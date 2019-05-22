import numpy as np
import torch
from PIL import Image
import NeuralNetModel
import torch
import numpy as np
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = img.crop((16, 16, 240, 240))
    np_img = np.array(img)
    np_img = np_img / 255
    mean_array = [0.485, 0.456, 0.406]
    std_array = [0.229, 0.224, 0.225]
    np_img = (np_img - mean_array) / std_array
    np_img = np_img.transpose(2, 0, 1)
    return np_img

def predict(image_path, check_point_path, category_names=None, topk=5, gpu=False):
    print("start predict:")
    model, optimizer_t, class_ids_Map = NeuralNetModel.loadModel(check_point_path)
    flower_class_dict = dict(zip(class_ids_Map.values(), class_ids_Map.keys()))
    if model is not None:
        if gpu:
            model.cuda()
        else:
            model.cpu()
        model.eval()
        with torch.no_grad():
            img_np = process_image(image_path)
            img_tensor = torch.from_numpy(img_np)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.type_as(torch.FloatTensor())
            if gpu:
                img_tensor = img_tensor.cuda()
            output = model(img_tensor)
            output_K, indices_K = output.topk(topk)
            ps_K = torch.exp(output_K)
            ps_K = ps_K.cpu().numpy()[0]
            indices_K = indices_K.cpu().numpy()[0]
            print("indices:",indices_K)
            flower_tpye = [flower_class_dict[index] for index in indices_K]
            if category_names is not None:
                flower_type_table = [category_names[key] for key in flower_tpye]
                return ps_K, flower_type_table
            else:
                return ps_K, flower_tpye

