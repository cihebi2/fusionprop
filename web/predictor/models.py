from django.db import models
import uuid
from django.core.exceptions import ValidationError

class PredictionJob(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    job_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, blank=True, null=True, help_text="Optional job name (e.g., filename or user input)")
    # Remove the single sequence field
    # sequence = models.TextField()
    # Add sequence count
    sequence_count = models.IntegerField(default=0, help_text="Number of sequences in this job")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    # Remove individual result fields
    # toxicity_result = models.TextField(blank=True, null=True)
    # thermostability_result = models.TextField(blank=True, null=True)
    # solubility_result = models.TextField(blank=True, null=True)
    # Add a field to store all results as a JSON list
    results = models.TextField(blank=True, null=True, help_text="JSON formatted list of prediction results for all sequences")
    error_message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        # Update string representation
        return f"Job {self.job_id} ({self.name or 'Unnamed'}) - {self.sequence_count} sequences - {self.status}"


# Visitor Counter Model
class SiteVisitCounter(models.Model):
    """Stores the total site visit count."""
    # We only ever need one row, but using a model is standard Django practice.
    count = models.PositiveBigIntegerField(default=0)

    # Optional: Make it a singleton to enforce only one instance
    def save(self, *args, **kwargs):
        if not self.pk and SiteVisitCounter.objects.exists():
            # Prevent creating a new instance if one already exists
            raise ValidationError('There can be only one SiteVisitCounter instance')
        return super(SiteVisitCounter, self).save(*args, **kwargs)

    def __str__(self):
        return f"Site Visit Count: {self.count}"

    class Meta:
        verbose_name_plural = "Site Visit Counter" # Nicer name in Django Admin
