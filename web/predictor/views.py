from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpRequest # Import HttpRequest for type hinting
from django.views.decorators.http import require_GET # For check_job_status
from .forms import PredictionForm
from .models import PredictionJob
from .tasks import run_prediction_task # Keep this import
import json
import uuid # Import uuid
from celery.result import AsyncResult
from django.core.paginator import Paginator # Import Paginator
import csv # Import the csv module
from django.http import HttpResponse # Ensure HttpResponse is imported

# Modify the submit_prediction view
def submit_prediction(request: HttpRequest):
    if request.method == 'POST':
        # Important: Pass request.FILES to the form constructor for file uploads
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            # Retrieve parsed data from the form's cleaned_data
            sequences = form.cleaned_data['parsed_sequences']
            names = form.cleaned_data['parsed_names']
            job_name = form.cleaned_data['name'] # This might be the filename or user input

            # --- Model Modification Needed Here ---
            # Before running this view, you'll need to modify the PredictionJob model:
            # 1. Remove the 'sequence' field.
            # 2. Add 'sequence_count = models.IntegerField()'
            # 3. Remove 'toxicity_result', 'thermostability_result', 'solubility_result'.
            # 4. Add 'results = models.TextField(blank=True, null=True)' (or JSONField)
            # 5. Run 'python manage.py makemigrations predictor' and 'python manage.py migrate'
            # --- End Model Modification Note ---

            # Create the database record for the batch job
            job = PredictionJob.objects.create(
                name=job_name,
                sequence_count=len(sequences), # Store the number of sequences
                status='PENDING'
            )

            # Launch the Celery task with the list of sequences and names
            run_prediction_task.delay(str(job.job_id), sequences, names)

            # Redirect to the result page for this job
            return redirect('predictor:job_list') 
        # If form is not valid, it will fall through and render the form with errors
    else:
        form = PredictionForm() # Empty form for GET request

    # Add example sequences for the template context
    example_sequences = """>Example_Seq1_Toxin
GFGCTLGKLEGCDGETCGADGWCDGPCQNGGSCHGGSGSIGTGCGSCKGKECVCR
>Example_Seq2_Soluble
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGVVFERGTHGSELKGHQVPGDWFATSDLSFMGSSGSKHAVFKSAADGHLARAGQSVPEFLSFKEQRTLAELALASAARFD"""

    return render(request, 'predictor/submit_form.html', {
        'form': form,
        'example_sequences': example_sequences
    })

# Modify the prediction_result view
def prediction_result(request: HttpRequest, job_id: uuid.UUID):
    job = get_object_or_404(PredictionJob, job_id=job_id)

    results_list = [] # Initialize empty list
    error_parsing = False

    # If the job is completed, try to parse the results JSON
    if job.status == 'COMPLETED' and job.results:
        try:
            results_list = json.loads(job.results)
            # Basic validation: Check if it's a list
            if not isinstance(results_list, list):
                raise json.JSONDecodeError("Results are not a list", job.results, 0)
        except json.JSONDecodeError as e:
            error_parsing = True
            print(f"Error parsing results JSON for job {job_id}: {e}")
            # Optionally update job status to FAILED here if parsing fails critically
            # job.status = 'FAILED'
            # job.error_message = "Internal error: Failed to parse prediction results."
            # job.save()

    context = {
        'job': job,
        'results_list': results_list, # Pass the list of results
        'error_parsing': error_parsing,
    }
    # Remove old context variables if they existed
    # 'toxicity_data': None,
    # 'thermostability_data': None,
    # 'solubility_data': None,

    return render(request, 'predictor/result_page.html', context)

# Modify check_job_status view (add type hint and decorator)
# Modify check_job_status view to include progress
@require_GET # Ensure this view only handles GET requests
def check_job_status(request: HttpRequest, job_id: uuid.UUID):
    """API endpoint for polling job status and progress."""
    job = get_object_or_404(PredictionJob, job_id=job_id)
    progress = None
    status_text = job.get_status_display() # Get display status from model

    # If the job is still processing according to the DB, check Celery task state for details
    if job.status == 'PROCESSING' or job.status == 'PENDING':
        try:
            task_result = AsyncResult(str(job.job_id)) # Get task result object using job_id
            if task_result.state == 'PROGRESS':
                progress = task_result.info # .info contains the 'meta' dictionary
                status_text = task_result.info.get('status', status_text) # Use status from meta if available
            elif task_result.state == 'PENDING':
                 status_text = 'Waiting for worker...'
            # Handle unexpected task states if necessary
            # elif task_result.state in ['FAILURE', 'SUCCESS', 'REVOKED']:
            #     # If task finished but DB not updated yet, maybe rely on DB status
            #     pass

        except Exception as e:
            # Handle cases where task result retrieval fails
            print(f"Could not retrieve Celery task state for job {job_id}: {e}")
            status_text = "Processing (status check error)"


    return JsonResponse({
        'status': job.status, # Still report DB status primarily for reload logic
        'job_id': str(job.job_id),
        'progress': progress, # This will be the meta dict (e.g., {'current': 5, 'total': 10, 'status': '...'}) or None
        'status_text': status_text # Provide more detailed status text
        })

# Modify the download_results view
@require_GET # Ensure this view only handles GET requests
def download_results(request: HttpRequest, job_id: uuid.UUID):
    """Downloads the prediction results as a CSV file."""
    job = get_object_or_404(PredictionJob, job_id=job_id)

    if job.status != 'COMPLETED' or not job.results:
        # Handle cases where results are not ready or available
        # Optionally, return an error message or redirect
        return HttpResponse("Results are not available for download.", status=404)

    try:
        results_list = json.loads(job.results)
        if not isinstance(results_list, list):
            raise ValueError("Stored results are not in the expected list format.")
    except (json.JSONDecodeError, ValueError) as e:
        # Handle error if results JSON is invalid
        return HttpResponse(f"Error parsing results data: {e}", status=500)

    # Create the HttpResponse object with the appropriate CSV header.
    # Use 'utf-8-sig' to include a BOM for better Excel compatibility
    response = HttpResponse(
        content_type='text/csv; charset=utf-8-sig',
        headers={'Content-Disposition': f'attachment; filename="prediction_results_{job_id}.csv"'},
    )

    writer = csv.writer(response)

    # Define the header row - ONLY include numerical/simple fields
    header = [
        '#',
        'Name',
        'Sequence',
        'Toxicity Probability',
        'Is Toxic', # Boolean (True/False)
        'Thermostability Value',
        'Solubility Value',
        'Status', # Simple status text (e.g., Success, Error)
        'Error Message'
    ]
    writer.writerow(header)

    # Write data rows
    for i, result in enumerate(results_list):
        row_num = i + 1
        name = result.get('name', 'N/A')
        # Use full sequence, assuming it's stored under the key 'sequence'
        # Fallback to 'sequence_preview' if 'sequence' is not found, then 'N/A'
        full_sequence = result.get('sequence', result.get('sequence_preview', 'N/A'))
        error_msg = result.get('error', '')

        if error_msg:
            # If there was an error for this sequence
            row = [
                row_num,
                name,
                full_sequence, # Use full sequence
                '', # Toxicity Prob
                '', # Is Toxic
                '', # Thermo Value
                '', # Sol Value
                'Error', # Status
                error_msg # Error Message
            ]
        else:
            # Extract numerical/boolean data, using .get() for safety
            toxicity_data = result.get('toxicity', {})
            thermo_data = result.get('thermostability', {})
            sol_data = result.get('solubility', {})

            row = [
                row_num,
                name,
                full_sequence, # Use full sequence
                toxicity_data.get('toxicity_probability'),
                toxicity_data.get('is_toxic'), # Keep boolean
                thermo_data.get('thermostability'),
                sol_data.get('solubility'),
                'Success', # Status
                '' # Error Message
            ]

        # Write the row to the CSV file
        writer.writerow(row)

    return response

def job_list(request: HttpRequest):
    """Displays a list of submitted prediction jobs."""
    # Fetch all jobs, ordered by creation date (newest first)
    all_jobs = PredictionJob.objects.all().order_by('-created_at')

    # --- Optional: Add Pagination ---
    paginator = Paginator(all_jobs, 25) # Show 25 jobs per page
    page_number = request.GET.get('page')
    jobs_page = paginator.get_page(page_number)
    # --- End Optional: Add Pagination ---

    context = {
        # 'jobs': all_jobs, # Use this if not using pagination
        'jobs_page': jobs_page, # Pass the paginated page object to the template
        'page_title': "Prediction Jobs", # Optional title for the template
    }
    return render(request, 'predictor/job_list.html', context)

def home_view(request: HttpRequest):
    """Renders the home page."""
    # You can add context data here if needed later
    context = {
        'page_title': 'Home',
    }
    return render(request, 'home.html', context)

def resource_view(request: HttpRequest):
    """Renders the resource page."""
    context = {
        'page_title': 'Resources',
    }
    return render(request, 'resource.html', context)

def about_view(request: HttpRequest):
    """Renders the about page."""
    context = {
        'page_title': 'About Us',
    }
    return render(request, 'about.html', context)

# Ensure necessary imports are present at the top
from django.http import HttpResponse, JsonResponse, HttpRequest
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_GET
from .forms import PredictionForm
from .models import PredictionJob
from .tasks import run_prediction_task
from celery.result import AsyncResult
import json
import uuid
import csv
from io import StringIO