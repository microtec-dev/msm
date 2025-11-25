import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .models import Student, Attendance, CameraConfiguration, ClassRoom, Timetable
from django.core.files.base import ContentFile
from datetime import datetime, timedelta
from django.utils import timezone
from django.utils.text import slugify
import pygame  # Import pygame for playing sounds
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required, user_passes_test
import threading
import time
import base64
from django.db import IntegrityError


# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Safe imread function to handle Unicode paths on Windows
def safe_imread(path):
    """Read image with fallback for Unicode paths."""
    img = cv2.imread(path)
    if img is None:
        try:
            # Fallback: read as bytes and decode
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Safe read failed for {path}: {e}")
            return None
    return img

# Function to detect and encode faces
def detect_and_encode(image, boxes=None):
    """Detect faces (or use provided boxes) and return embeddings list."""
    with torch.no_grad():
        if boxes is None:
            boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                try:
                    face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                except Exception:
                    continue
                if face is None or face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0)
                encoding = resnet(face_tensor).detach().numpy().flatten()
                faces.append(encoding)
            return faces
    return []

# Function to encode uploaded images
def encode_uploaded_images():
    known_face_encodings = []
    known_face_names = []

    # Fetch only authorized images
    uploaded_images = Student.objects.filter(authorized=True)

    for student in uploaded_images:
        image_path = os.path.join(settings.MEDIA_ROOT, str(student.image))
        known_image = safe_imread(image_path)
        # Skip if image failed to load
        if known_image is None or known_image.size == 0:
            print(f"Warning: Could not read image for student {student.name} at {image_path}")
            continue
        try:
            known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"OpenCV error converting image for {student.name}: {e}")
            continue
        encodings = detect_and_encode(known_image_rgb)
        if encodings:
            # Map each encoding to the student name
            for encoding in encodings:
                known_face_encodings.append(encoding)
                known_face_names.append(student.name)
    
    print(f"Loaded {len(known_face_encodings)} face encodings from {len(uploaded_images)} authorized students")
    return known_face_encodings, known_face_names

# Function to recognize faces
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.7):
    """Recognize faces with adjustable threshold (lower = stricter, higher = more lenient)."""
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        if min_distance < threshold:
            recognized_names.append(known_names[min_distance_idx])
            print(f"Recognized: {known_names[min_distance_idx]} (distance: {min_distance:.3f})")
        else:
            recognized_names.append('Not Recognized')
            print(f"Not recognized (closest: {known_names[min_distance_idx]}, distance: {min_distance:.3f}, threshold: {threshold})")
    return recognized_names

# View for capturing student information and image
def capture_student(request):
    classes = []
    try:
        from .models import ClassRoom
        classes = ClassRoom.objects.all()
    except Exception:
        classes = []

    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        classroom_id = request.POST.get('student_class')
        image_data = request.POST.get('image_data')

        # Decode the base64 image data
        if image_data:
            header, encoded = image_data.split(',', 1)
            # Slugify filename to avoid Unicode issues
            safe_filename = slugify(name) + '.jpg'
            image_file = ContentFile(base64.b64decode(encoded), name=safe_filename)

            student = Student(
                name=name,
                email=email,
                phone_number=phone_number,
                student_class='',
                image=image_file,
                authorized=False  # Default to False during registration
            )
            # assign classroom if provided
            if classroom_id:
                try:
                    from .models import ClassRoom
                    student.classroom = ClassRoom.objects.get(pk=int(classroom_id))
                except Exception:
                    pass
            
            student.save()

            return redirect('selfie_success')  # Redirect to a success page

    return render(request, 'capture_student.html', {'classes': classes})


# Success view after capturing student information and image
def selfie_success(request):
    return render(request, 'selfie_success.html')


# This views for capturing studen faces and recognize
def capture_and_recognize(request):
    stop_events = []  # List to store stop events for each thread
    camera_threads = []  # List to store threads for each camera
    camera_windows = []  # List to store window names
    error_messages = []  # List to capture errors from threads

    def process_frame(cam_config, stop_event):
        """Thread function to capture and process frames for each camera."""
        cap = None
        window_created = False  # Flag to track if the window was created
        try:
            # Check if the camera source is a number (local webcam) or a string (IP camera URL)
            if cam_config.camera_source.isdigit():
                cap = cv2.VideoCapture(int(cam_config.camera_source))  # Use integer index for webcam
            else:
                cap = cv2.VideoCapture(cam_config.camera_source)  # Use string for IP camera URL

            if not cap.isOpened():
                raise Exception(f"Unable to access camera {cam_config.name}.")

            threshold = cam_config.threshold

            # Initialize pygame mixer for sound playback
            pygame.mixer.init()
            success_sound = pygame.mixer.Sound('app/suc.wav')  # load sound path

            window_name = f'Face Recognition - {cam_config.name}'
            camera_windows.append(window_name)  # Track the window name
            
            consecutive_failures = 0
            max_failures = 5  # Allow 5 consecutive failures before giving up

            while not stop_event.is_set():

                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    print(f"Failed to capture frame for camera: {cam_config.name} (attempt {consecutive_failures}/{max_failures})")
                    if consecutive_failures >= max_failures:
                        print(f"Too many failures, stopping camera: {cam_config.name}")
                        break
                    time.sleep(0.1)  # Brief pause before retry
                    continue  # Retry instead of breaking immediately

                consecutive_failures = 0  # Reset counter on successful read

                # Schedule check: if schedule_enabled and class assigned, verify current time is within any timetable window
                try:
                    now_dt = datetime.now()
                    is_active = True
                    if cam_config.schedule_enabled and cam_config.class_assigned:
                        from .models import Timetable
                        weekday = now_dt.weekday()  # Monday=0
                        timetables = Timetable.objects.filter(classroom=cam_config.class_assigned, day_of_week=weekday)
                        is_active = False
                        for tt in timetables:
                            pre = tt.pre_minutes if tt.pre_minutes is not None else cam_config.pre_minutes
                            post = tt.post_minutes if tt.post_minutes is not None else cam_config.post_minutes
                            start_dt = datetime.combine(now_dt.date(), tt.start_time) - timedelta(minutes=pre or 0)
                            end_dt = datetime.combine(now_dt.date(), tt.end_time) + timedelta(minutes=post or 0)
                            if start_dt <= now_dt <= end_dt:
                                is_active = True
                                break
                    if not is_active:
                        # sleep briefly and skip processing heavy detection
                        time.sleep(0.5)
                        continue
                except Exception as e:
                    print(f"Schedule check error for {cam_config.name}: {e}")

                # Motion detection: run cheap check before expensive face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                if not hasattr(process_frame, 'prev_gray'):
                    process_frame.prev_gray = {}
                prev = process_frame.prev_gray.get(cam_config.name)
                motion_detected = True
                if prev is None:
                    process_frame.prev_gray[cam_config.name] = gray
                else:
                    frame_delta = cv2.absdiff(prev, gray)
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    non_zero = np.count_nonzero(thresh)
                    # heuristic threshold (tune as needed)
                    motion_threshold = 2000
                    motion_detected = non_zero > motion_threshold
                    process_frame.prev_gray[cam_config.name] = gray

                if not motion_detected:
                    # skip heavy face detection
                    if not window_created:
                        cv2.namedWindow(window_name)
                        window_created = True
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break
                    continue

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # First detect boxes to avoid double detection
                boxes, _ = mtcnn.detect(frame_rgb)
                if boxes is None:
                    # No faces detected
                    if not window_created:
                        cv2.namedWindow(window_name)
                        window_created = True
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break
                    continue

                test_face_encodings = detect_and_encode(frame_rgb, boxes=boxes)

                if test_face_encodings:
                    known_face_encodings, known_face_names = encode_uploaded_images()  # Load known face encodings once
                    if known_face_encodings:
                        names = recognize_faces(np.array(known_face_encodings), known_face_names, test_face_encodings, threshold)

                        # debounce settings: avoid toggling attendance too often if person walks by repeatedly
                        recognition_cooldown_seconds = 300  # 5 minutes default cooldown per student
                        min_checkout_seconds = 60  # require at least 60s after check-in before allowing check-out
                        if not hasattr(process_frame, 'last_seen'):
                            process_frame.last_seen = {}

                        for name, box in zip(names, boxes):
                            if box is not None:
                                (x1, y1, x2, y2) = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                                if name != 'Not Recognized':
                                    students = Student.objects.filter(name=name)
                                    if students.exists():
                                        student = students.first()

                                        # Manage attendance with debounce to prevent rapid toggles
                                        attendance, created = Attendance.objects.get_or_create(student=student, date=datetime.now().date())
                                        now_tz = timezone.now()
                                        last_seen = process_frame.last_seen.get(student.pk)

                                        # If we've seen this student recently, skip toggling (avoid noise when walking by)
                                        if last_seen and (now_tz - last_seen).total_seconds() < recognition_cooldown_seconds:
                                            # Annotate as recently seen and continue
                                            cv2.putText(frame, f"{name} (recent)", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                                            continue

                                        if created:
                                            attendance.mark_checked_in()
                                            process_frame.last_seen[student.pk] = now_tz
                                            success_sound.play()
                                            cv2.putText(frame, f"{name}, checked in.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                        else:
                                            if attendance.check_in_time and not attendance.check_out_time:
                                                # allow check-out only after a minimum duration since check-in
                                                if now_tz >= attendance.check_in_time + timedelta(seconds=min_checkout_seconds):
                                                    attendance.mark_checked_out()
                                                    process_frame.last_seen[student.pk] = now_tz
                                                    success_sound.play()
                                                    cv2.putText(frame, f"{name}, checked out.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                                else:
                                                    cv2.putText(frame, f"{name}, checked in.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                            elif attendance.check_in_time and attendance.check_out_time:
                                                # already checked out for today — do not toggle back in immediately
                                                cv2.putText(frame, f"{name}, checked out.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Display frame in separate window for each camera
                if not window_created:
                    cv2.namedWindow(window_name)  # Only create window once
                    window_created = True  # Mark window as created
                
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()  # Signal the thread to stop when 'q' is pressed
                    break

        except Exception as e:
            print(f"Error in thread for {cam_config.name}: {e}")
            error_messages.append(str(e))  # Capture error message
        finally:
            if cap is not None:
                cap.release()
            if window_created:
                cv2.destroyWindow(window_name)  # Only destroy if window was created

    try:
        # Get all camera configurations
        cam_configs = CameraConfiguration.objects.all()
        if not cam_configs.exists():
            raise Exception("No camera configurations found. Please configure them in the admin panel.")

        # Create threads for each camera configuration
        for cam_config in cam_configs:
            stop_event = threading.Event()
            stop_events.append(stop_event)

            camera_thread = threading.Thread(target=process_frame, args=(cam_config, stop_event))
            camera_threads.append(camera_thread)
            camera_thread.start()

        # Keep the main thread running while cameras are being processed
        while any(thread.is_alive() for thread in camera_threads):
            time.sleep(1)  # Non-blocking wait, allowing for UI responsiveness

    except Exception as e:
        error_messages.append(str(e))  # Capture the error message
    finally:
        # Ensure all threads are signaled to stop
        for stop_event in stop_events:
            stop_event.set()

        # Ensure all windows are closed in the main thread
        for window in camera_windows:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:  # Check if window exists
                cv2.destroyWindow(window)

    # Check if there are any error messages
    if error_messages:
        # Join all error messages into a single string
        full_error_message = "\n".join(error_messages)
        return render(request, 'error.html', {'error_message': full_error_message})  # Render the error page with message

    return redirect('student_attendance_list')

#this is for showing Attendance list
def student_attendance_list(request):
    # Get the search query and date filter from the request
    search_query = request.GET.get('search', '')
    date_filter = request.GET.get('attendance_date', '')

    # Get all students
    students = Student.objects.all()

    # Filter students based on the search query
    if search_query:
        students = students.filter(name__icontains=search_query)

    # Prepare the attendance data
    student_attendance_data = []

    for student in students:
        # Get the attendance records for each student, filtering by attendance date if provided
        attendance_records = Attendance.objects.filter(student=student)

        if date_filter:
            # Assuming date_filter is in the format YYYY-MM-DD
            attendance_records = attendance_records.filter(date=date_filter)

        attendance_records = attendance_records.order_by('date')
        
        student_attendance_data.append({
            'student': student,
            'attendance_records': attendance_records
        })

    context = {
        'student_attendance_data': student_attendance_data,
        'search_query': search_query,  # Pass the search query to the template
        'date_filter': date_filter       # Pass the date filter to the template
    }
    return render(request, 'student_attendance_list.html', context)


def home(request):
    return render(request, 'home.html')


# Custom user pass test for admin access
def is_admin(user):
    return user.is_superuser

@login_required
@user_passes_test(is_admin)
def student_list(request):
    students = Student.objects.all()
    return render(request, 'student_list.html', {'students': students})

@login_required
@user_passes_test(is_admin)
def student_detail(request, pk):
    student = get_object_or_404(Student, pk=pk)
    return render(request, 'student_detail.html', {'student': student})


@login_required
@user_passes_test(is_admin)
def student_update(request, pk):
    student = get_object_or_404(Student, pk=pk)

    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        student_class = request.POST.get('student_class')
        authorized = request.POST.get('authorized') == 'on'

        # Update fields
        student.name = name
        student.email = email
        student.phone_number = phone_number
        student.student_class = student_class
        student.authorized = authorized

        # Handle image upload
        if 'image' in request.FILES and request.FILES['image']:
            student.image = request.FILES['image']

        # handle classroom selection
        classroom_id = request.POST.get('student_class')
        if classroom_id:
            try:
                from .models import ClassRoom
                student.classroom = ClassRoom.objects.get(pk=int(classroom_id))
            except Exception:
                student.classroom = None

        student.save()
        messages.success(request, 'Student updated successfully.')
        return redirect('student-detail', pk=student.pk)

    # supply classes for select
    try:
        classes = ClassRoom.objects.all()
    except Exception:
        classes = []
    return render(request, 'student_form.html', {'student': student, 'classes': classes})

@login_required
@user_passes_test(is_admin)
def student_authorize(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        authorized = request.POST.get('authorized', False)
        student.authorized = bool(authorized)
        student.save()
        return redirect('student-detail', pk=pk)
    
    return render(request, 'student_authorize.html', {'student': student})

# This views is for Deleting student
@login_required
@user_passes_test(is_admin)
def student_delete(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        student.delete()
        messages.success(request, 'Student deleted successfully.')
        return redirect('student-list')  # Redirect to the student list after deletion
    
    return render(request, 'student_delete_confirm.html', {'student': student})


# View function for user login
def user_login(request):
    # Check if the request method is POST, indicating a form submission
    if request.method == 'POST':
        # Retrieve username and password from the submitted form data
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Authenticate the user using the provided credentials
        user = authenticate(request, username=username, password=password)

        # Check if the user was successfully authenticated
        if user is not None:
            # Log the user in by creating a session
            login(request, user)
            # Redirect the user to the student list page after successful login
            return redirect('home')  # Replace 'student-list' with your desired redirect URL after login
        else:
            # If authentication fails, display an error message
            messages.error(request, 'Invalid username or password.')

    # Render the login template for GET requests or if authentication fails
    return render(request, 'login.html')


# This is for user logout
def user_logout(request):
    logout(request)
    return redirect('login')  # Replace 'login' with your desired redirect URL after logout

# Function to handle the creation of a new camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_create(request):
    # Check if the request method is POST, indicating form submission
    # load classes for select
    try:
        classes = ClassRoom.objects.all()
    except Exception:
        classes = []

    if request.method == "POST":
        # Retrieve form data from the request
        name = request.POST.get('name')
        camera_source = request.POST.get('camera_source')
        threshold = request.POST.get('threshold')
        motion_threshold = request.POST.get('motion_threshold')
        class_assigned = request.POST.get('class_assigned') or request.POST.get('student_class')
        schedule_enabled = request.POST.get('schedule_enabled') == 'on'
        pre_minutes = request.POST.get('pre_minutes')
        post_minutes = request.POST.get('post_minutes')

        try:
            # Save the data to the database using the CameraConfiguration model
            cfg = CameraConfiguration.objects.create(
                name=name,
                camera_source=camera_source,
                threshold=threshold,
                motion_threshold=motion_threshold or 2000,
                schedule_enabled=schedule_enabled,
                pre_minutes=int(pre_minutes) if pre_minutes else 5,
                post_minutes=int(post_minutes) if post_minutes else 5,
            )
            if class_assigned:
                try:
                    cfg.class_assigned = ClassRoom.objects.get(pk=int(class_assigned))
                    cfg.save()
                except Exception:
                    pass
            # Redirect to the list of camera configurations after successful creation
            return redirect('camera_config_list')

        except IntegrityError:
            # Handle the case where a configuration with the same name already exists
            messages.error(request, "A configuration with this name already exists.")
            # Render the form again to allow user to correct the error
            return render(request, 'camera_config_form.html')

    # Render the camera configuration form for GET requests
    return render(request, 'camera_config_form.html', {'classes': classes})


# READ: Function to list all camera configurations
@login_required
@user_passes_test(is_admin)
def camera_config_list(request):
    # Retrieve all CameraConfiguration objects from the database
    configs = CameraConfiguration.objects.all()
    # Render the list template with the retrieved configurations
    return render(request, 'camera_config_list.html', {'configs': configs})


# UPDATE: Function to edit an existing camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_update(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating form submission
    # load classes for select
    try:
        classes = ClassRoom.objects.all()
    except Exception:
        classes = []

    if request.method == "POST":
        # Update the configuration fields with data from the form
        config.name = request.POST.get('name')
        config.camera_source = request.POST.get('camera_source')
        config.threshold = request.POST.get('threshold')
        config.success_sound_path = request.POST.get('success_sound_path')
        # new fields
        config.motion_threshold = int(request.POST.get('motion_threshold') or config.motion_threshold)
        class_assigned = request.POST.get('class_assigned') or request.POST.get('student_class')
        if class_assigned:
            try:
                config.class_assigned = ClassRoom.objects.get(pk=int(class_assigned))
            except Exception:
                config.class_assigned = None
        config.schedule_enabled = request.POST.get('schedule_enabled') == 'on'
        config.pre_minutes = int(request.POST.get('pre_minutes') or config.pre_minutes)
        config.post_minutes = int(request.POST.get('post_minutes') or config.post_minutes)

        # Save the changes to the database
        config.save()  

        # Redirect to the list page after successful update
        return redirect('camera_config_list')  
    
    # Render the configuration form with the current configuration data for GET requests
    return render(request, 'camera_config_form.html', {'config': config, 'classes': classes})


# DELETE: Function to delete a camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_delete(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating confirmation of deletion
    if request.method == "POST":
        # Delete the record from the database
        config.delete()  
        # Redirect to the list of camera configurations after deletion
        return redirect('camera_config_list')

    # Render the delete confirmation template with the configuration data
    return render(request, 'camera_config_delete.html', {'config': config})


# Class and Timetable management views
@login_required
@user_passes_test(is_admin)
def class_list(request):
    classes = ClassRoom.objects.all()
    return render(request, 'class_list.html', {'classes': classes})


@login_required
@user_passes_test(is_admin)
def class_create(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        if name:
            ClassRoom.objects.create(name=name)
            return redirect('class_list')
    return render(request, 'class_form.html')


@login_required
@user_passes_test(is_admin)
def class_update(request, pk):
    cls = get_object_or_404(ClassRoom, pk=pk)
    if request.method == 'POST':
        cls.name = request.POST.get('name')
        cls.save()
        return redirect('class_list')
    return render(request, 'class_form.html', {'cls': cls})


@login_required
@user_passes_test(is_admin)
def class_delete(request, pk):
    cls = get_object_or_404(ClassRoom, pk=pk)
    if request.method == 'POST':
        cls.delete()
        return redirect('class_list')
    return render(request, 'class_delete_confirm.html', {'cls': cls})


@login_required
@user_passes_test(is_admin)
def timetable_list(request, pk):
    classroom = get_object_or_404(ClassRoom, pk=pk)
    timetables = Timetable.objects.filter(classroom=classroom).order_by('day_of_week', 'start_time')
    return render(request, 'timetable_list.html', {'classroom': classroom, 'timetables': timetables})


@login_required
@user_passes_test(is_admin)
def timetable_create(request, pk):
    classroom = get_object_or_404(ClassRoom, pk=pk)
    if request.method == 'POST':
        day = int(request.POST.get('day_of_week'))
        start = request.POST.get('start_time')
        end = request.POST.get('end_time')
        pre = request.POST.get('pre_minutes') or None
        post = request.POST.get('post_minutes') or None
        tt = Timetable.objects.create(
            classroom=classroom,
            day_of_week=day,
            start_time=start,
            end_time=end,
            pre_minutes=int(pre) if pre else None,
            post_minutes=int(post) if post else None,
        )
        return redirect('timetable_list', pk=classroom.pk)
    return render(request, 'timetable_form.html', {'classroom': classroom})


@login_required
@user_passes_test(is_admin)
def timetable_delete(request, pk):
    tt = get_object_or_404(Timetable, pk=pk)
    classroom_pk = tt.classroom.pk
    if request.method == 'POST':
        tt.delete()
        return redirect('timetable_list', pk=classroom_pk)
    return render(request, 'timetable_delete_confirm.html', {'timetable': tt, 'classroom': tt.classroom})
