import torch
print(torch.__version__)  # Должно быть `2.x.x+cu118` или `2.x.x+cu121`
print(torch.cuda.is_available())  # Должно быть `True`
print(torch.cuda.get_device_name(0))  # Название вашей видеокарты