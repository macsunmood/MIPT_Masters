CLASS_LABELS = {
    'Background' : (0, 0, 0), 
    'Field'      : (0.294, 0, 0.51), 
    'Forest'     : (0.7, 0.1, 0.1), 
    'Grass'      : (0.4, 0.6, 1), 
    'Power lines': (1, 0, 0), 
    'Road'       : (1, 0.749, 0),
    'Water'      : (0.3, 0.802, 0.3)
}

CLASS_LABELS = {
    name: tuple(int(ch * 255) for ch in color) 
    for name, color in CLASS_LABELS.items()
}


MODELS_GDRIVE = {
    'best_detect_YOLO8.pt':                       'https://drive.google.com/uc?id=1nQs75C05BChBUl_dh7hIDgiZl9Wb5Kvi',  #
    'psp_unet_model_best__12.10.2023.h5':         'https://drive.google.com/uc?id=1p8uC1zW6LAdtkePH_FbUgbFA7a3IPbqv',  #
}


AUDIOFILES_GDRIVE = {
    'temp':  'https://drive.google.com/uc?id=16wDzvHhc165dg7pGXtOuEVAST9Fzqmic',  #
    'temp2': 'https://drive.google.com/uc?id=1JUnXrS2goXXOxljCNA3QGXLXUn6opUv6',  #
}


EXAMPLES = {
    'Test 1: Forest + Road': {
        'url':    'https://www.youtube.com/watch?v=5LAgrI-hH2c', 
        'start':  0.0
    }, 
    'Test 2: Fields': {
        'url':    'https://www.youtube.com/watch?v=04z02TNjio0', 
        'start':  0.0
    }, 
    
    'Test 3: Forest': {
        'url':    'https://www.youtube.com/watch?v=Woo-9cduWiE', 
        'start':  0.0
    }
}
