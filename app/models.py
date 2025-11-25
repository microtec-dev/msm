from django.db import models
from django.utils import timezone

class Student(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField(max_length=255)
    phone_number = models.CharField(max_length=15)
    student_class = models.CharField(max_length=100)
    # New relation to ClassRoom (optional for backward compatibility)
    classroom = models.ForeignKey('ClassRoom', null=True, blank=True, on_delete=models.SET_NULL)
    image = models.ImageField(upload_to='students/')
    authorized = models.BooleanField(default=False)

    def __str__(self):
        return self.name

class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField()
    check_in_time = models.DateTimeField(null=True, blank=True)
    check_out_time = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.student.name} - {self.date}"

    def mark_checked_in(self):
        self.check_in_time = timezone.now()
        self.save()

    def mark_checked_out(self):
        if self.check_in_time:
            self.check_out_time = timezone.now()
            self.save()
        else:
            raise ValueError("Cannot mark check-out without check-in.")

    def calculate_duration(self):
        if self.check_in_time and self.check_out_time:
            duration = self.check_out_time - self.check_in_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        return None

    def save(self, *args, **kwargs):
        if not self.pk:  # Only on creation
            self.date = timezone.now().date()
        super().save(*args, **kwargs)




class CameraConfiguration(models.Model):
    name = models.CharField(max_length=100, unique=True, help_text="Give a name to this camera configuration")
    camera_source = models.CharField(max_length=255, help_text="Camera index (0 for default webcam or RTSP/HTTP URL for IP camera)")
    threshold = models.FloatField(default=0.8, help_text="Face recognition confidence threshold (0.7-0.9 recommended)")
    # Optional: assign camera to a class so schedule can control it
    class_assigned = models.ForeignKey('ClassRoom', null=True, blank=True, on_delete=models.SET_NULL)
    # Schedule-related offsets in minutes
    pre_minutes = models.IntegerField(default=5, help_text='Activate camera N minutes before timetable start')
    post_minutes = models.IntegerField(default=5, help_text='Keep camera active N minutes after timetable end')
    schedule_enabled = models.BooleanField(default=False, help_text='Enable schedule-based activation')
    # Motion detection threshold (number of changed pixels)
    motion_threshold = models.IntegerField(default=2000, help_text='Number of changed pixels to consider motion detected')

    def __str__(self):
        return self.name


class ClassRoom(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name


class Timetable(models.Model):
    DAYS = [
        (0, 'Monday'),
        (1, 'Tuesday'),
        (2, 'Wednesday'),
        (3, 'Thursday'),
        (4, 'Friday'),
        (5, 'Saturday'),
        (6, 'Sunday'),
    ]
    classroom = models.ForeignKey(ClassRoom, on_delete=models.CASCADE)
    day_of_week = models.IntegerField(choices=DAYS)
    start_time = models.TimeField()
    end_time = models.TimeField()
    # Optional per-timetable pre/post offsets (overrides camera defaults if set)
    pre_minutes = models.IntegerField(null=True, blank=True)
    post_minutes = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"{self.classroom.name} - {self.get_day_of_week_display()} {self.start_time}-{self.end_time}"
