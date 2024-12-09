from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from .forms import RegisterForm, LoginForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import Joblisting
from ai_model.services import predict_job_listings
from django.http import HttpResponse
import pandas as pd
from django.db.models import Q
from io import StringIO

def landing(request):
    return render(request, 'landing.html')

@login_required
def home(request):
    return render(request, 'home.html')

def login_view(request):
    return render(request, 'login.html')

def register(request):
    return render(request, 'register.html')

@login_required
def search(request):
    q = request.GET.get('q') if request.GET.get('q') != None else ''

    # Get initial queryset from database search
    listings = Joblisting.objects.filter(
        Q(Titleicontains=q) |
        Q(Companyicontains=q) |
        Q(Locationicontains=q) |
        Q(JobRequirementicontains=q) |
        Q(RequiredQualicontains=q) |
        Q(Salaryicontains=q)
    )

    # Get all data including id (for our reference) and jobpost
    full_df = pd.DataFrame.from_records(
        listings.values(
            'id', 'jobpost', 'Title', 'Company', 'StartDate', 'Duration', 'Location',
            'JobDescription', 'JobRequirment', 'RequiredQual', 
            'Salary', 'AboutC'
        )
    )

    if not full_df.empty:
        # Create a new DataFrame with only the columns the model expects
        model_df = full_df[['jobpost', 'Title', 'Company', 'StartDate', 'Duration', 'Location',
                           'JobDescription', 'JobRequirment', 'RequiredQual',
                            'Salary', 'AboutC']]

    # Convert DataFrame to CSV string
    csv_buffer = StringIO()
    model_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    # Get predictions from AI model
    result_csv = predict_job_listings(csv_data)

    # Convert result CSV back to DataFrame
    result_df = pd.read_csv(StringIO(result_csv))

    # Update Prediction field in database for each listing
    for index, row in result_df.iterrows():
        listing_id = full_df.iloc[index]['id'] # Convert to string since Prediction field is CharField
        prediction = str(row['prediction'])

    Joblisting.objects.filter(id=listing_id).update(Prediction=prediction)

    # Get fresh listings from database with updated predictions
    updated_listings = Joblisting.objects.filter(
        Q(Titleicontains=q) |
        Q(Companyicontains=q) |
        Q(Locationicontains=q) |
        Q(JobRequirementicontains=q) |
        Q(RequiredQualicontains=q) |
        Q(Salaryicontains=q)
    )

    context = {'listings': updated_listings}
    return render(request, 'search.html', context)
#    q = request.GET.get('q') if request.GET.get('q') != None else ''

    # Get initial queryset from database search
#    listings = Joblisting.objects.filter(
#        Q(Titleicontains=q) |
#        Q(Companyicontains=q) |
#        Q(Locationicontains=q) |
#        Q(JobRequirementicontains=q) |
#        Q(RequiredQualicontains=q) |
#        Q(Salaryicontains=q)
#    )
#    context = {'listings': listings}


    return render(request, 'search.html', context)

def terms_service(request):
    return render(request, 'terms_service.html')

@login_required
def profiles(request):
    return render(request, 'profiles.html')

def main(request):
    return render(request, 'main.html')


def register_view(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save
            messages.success(request, "Registration successful!")
            return redirect('home')
        else:
            messages.error(request, "Registration failed. Please try again.")
    else:
        form = RegisterForm()  
    return render(request, 'register.html', {'form': form})

def login_view(request):
    if request.method == "POST":
        form = LoginForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, "Login successful!")
            return redirect('home')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.info(request, "Logged out successfully!")
    return redirect('login')

def home_view(request):
    return render(request, 'home.html')

def job_list(request):
    jobs = Joblisting.objects.all()
    return render(request, 'search.html', {'jobs': jobs})