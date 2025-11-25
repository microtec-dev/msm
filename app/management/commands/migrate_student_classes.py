from django.core.management.base import BaseCommand
from app.models import Student, ClassRoom

class Command(BaseCommand):
    help = 'Migrate existing student_class string values into ClassRoom records and assign to student.classroom'

    def handle(self, *args, **options):
        students = Student.objects.exclude(student_class__isnull=True).exclude(student_class__exact='')
        created = 0
        assigned = 0
        for s in students:
            cls_name = s.student_class.strip()
            if not cls_name:
                continue
            room, created_now = ClassRoom.objects.get_or_create(name=cls_name)
            if created_now:
                created += 1
                self.stdout.write(self.style.SUCCESS(f'Created ClassRoom: {room.name}'))
            s.classroom = room
            s.student_class = ''
            s.save()
            assigned += 1
        self.stdout.write(self.style.SUCCESS(f'Assigned {assigned} students. Created {created} classrooms.'))
