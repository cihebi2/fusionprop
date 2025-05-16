from django.db import models
from django.db.models import F # Important for atomic updates
from .models import SiteVisitCounter

def visitor_counter(request):
    """
    Increments the site visit count and adds it to the template context.
    """
    # Use get_or_create to handle the very first visit when the object doesn't exist
    counter_obj, created = SiteVisitCounter.objects.get_or_create(
        pk=1, # Assuming we always use the row with primary key 1
        defaults={'count': 0}
    )

    # Increment the count atomically to prevent race conditions
    # This updates the count directly in the database
    SiteVisitCounter.objects.filter(pk=1).update(count=F('count') + 1)

    # Retrieve the updated value to pass to the template
    # (We query again because .update() doesn't return the updated object directly)
    updated_counter = SiteVisitCounter.objects.get(pk=1)

    return {
        'visitor_count': updated_counter.count
    } 