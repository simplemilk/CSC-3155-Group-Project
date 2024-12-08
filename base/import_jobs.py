import csv
from base.models import JobPost

def run():
    with open('data/mockjobdataset.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            print(f"Importing job: {row['title']} at {row['company']}")
            JobPost.objects.create(
                title=row['title'],
                company=row['company'],
                start_date=row['start_date'] if row['start_date'] else None,
                duration=row['duration'],
                location=row['location'],
                description=row['description'],
                requirements=row['requirements'],
                qualifications=row['qualifications'],
                salary=row['salary'],
                about_company=row['about_company'],
                offers=row['offers']
            )