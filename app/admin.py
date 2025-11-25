from django.contrib import admin
from .models import Student, Attendance, CameraConfiguration, ClassRoom, Timetable


class TimetableInline(admin.TabularInline):
    model = Timetable
    extra = 1


@admin.register(ClassRoom)
class ClassRoomAdmin(admin.ModelAdmin):
    list_display = ['name']
    inlines = [TimetableInline]


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'phone_number', 'student_class', 'classroom', 'authorized']
    list_filter = ['student_class', 'authorized', 'classroom']
    search_fields = ['name', 'email']


@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ['student', 'date', 'check_in_time', 'check_out_time']
    list_filter = ['date']
    search_fields = ['student__name']

    def get_readonly_fields(self, request, obj=None):
        if obj:  # Editing an existing object
            return ['student', 'date', 'check_in_time', 'check_out_time']
        else:  # Adding a new object
            return ['date', 'check_in_time', 'check_out_time']

    def save_model(self, request, obj, form, change):
        if change:  # Editing an existing object
            # Ensure check-in and check-out times cannot be modified via admin
            obj.check_in_time = Attendance.objects.get(id=obj.id).check_in_time
            obj.check_out_time = Attendance.objects.get(id=obj.id).check_out_time
        super().save_model(request, obj, form, change)


@admin.register(CameraConfiguration)
class CameraConfigurationAdmin(admin.ModelAdmin):
    list_display = ['name', 'camera_source', 'threshold', 'schedule_enabled', 'class_assigned']
    list_filter = ['schedule_enabled', 'class_assigned']
    search_fields = ['name']