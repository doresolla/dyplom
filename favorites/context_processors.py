from .favorites import Favorites

def favorites_processor(request):
    return {'favorites_count': len(Favorites(request))}