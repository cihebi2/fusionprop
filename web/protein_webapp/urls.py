from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
# Import the views from the predictor app (or wherever you placed them)
from predictor import views as predictor_views
# Add these imports for static files during development
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),

    # --- Static Pages ---
    # Map the root URL to the home_view
    path('', predictor_views.home_view, name='home'),
    path('resource/', predictor_views.resource_view, name='resource'),
    path('about/', predictor_views.about_view, name='about'),

    # --- Predictor App URLs ---
    # Include predictor app's URLs under the '/predictor/' prefix
    # Use namespace='predictor' so {% url 'predictor:...' %} works
    path('predictor/', include(('predictor.urls', 'predictor'), namespace='predictor')),

    # Remove the old root redirect if you now have a dedicated home view
    # path('', lambda request: redirect('predictor/', permanent=False)),
]

# Add this pattern for serving static files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    # The following line is often used, but STATICFILES_DIRS handles finding files
    # in development without needing STATIC_ROOT defined yet.
    # If you collectstatic, STATIC_ROOT will be used in production.
    # We primarily need this to find files specified in STATICFILES_DIRS:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.BASE_DIR / "static")