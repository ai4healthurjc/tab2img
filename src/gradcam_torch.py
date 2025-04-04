import argparse
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import coloredlogs
import logging
from torchvision.utils import save_image
import torch.nn.functional as F
import shutil
import utils.consts as cons
import utils.loader as loader
import new_cnn_models as cnn
import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
from utils.dataset_creation import create_datasets

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def process_and_generate_combined_heatmaps(test_loader, test_loader_inter, target_layer, gradcam_filepath, model,
                                           device):
    """
    Procesa imágenes desde test_loader, genera mapas de calor, y los combina con las imágenes de test_loader_inter.
    """
    model.eval()

    for i, (image, _) in enumerate(test_loader):
        image = image.to(device)

        outputs = model(image)
        predicted_class = torch.sigmoid(outputs) > 0.5
        target_class = int(predicted_class.item())
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = cam(input_tensor=image, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

        image_inter = test_loader_inter.dataset[i][0]
        interpretability_image = image_inter.cpu().numpy().transpose(1, 2, 0)
        grayscale_cam = cv2.resize(grayscale_cam, (interpretability_image.shape[1], interpretability_image.shape[0]))

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        combined_image = cv2.addWeighted(interpretability_image, 0.6, heatmap_colored, 0.4, 0)

        class_folder = os.path.join(gradcam_filepath, str(target_class))
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        image_filename = f"heatmap_{i}.png"
        cv2.imwrite(os.path.join(class_folder, image_filename), combined_image)

    print("Mapas de calor generados y combinados con las imágenes de test_loader_inter.")


def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


def modify_and_create_path(original_path, old_folder, new_folder):
    """
    Modifica una parte de la ruta del archivo y asegura que la carpeta de destino exista.

    Args:
        original_path (str): Ruta original del archivo.
        old_folder (str): Nombre de la carpeta a reemplazar.
        new_folder (str): Nombre de la nueva carpeta.

    Returns:
        str: Nueva ruta con la carpeta modificada.
    """
    new_path = original_path.replace(old_folder, new_folder)

    new_dir = os.path.dirname(new_path)
    os.makedirs(new_dir, exist_ok=True)

    return new_path


class GradCAM(object):
    """
    Clase para calcular mapas GradCAM para un modelo PyTorch genérico.
    """

    def __init__(self, model, target_layer_name, verbose=False):
        """
        Inicializa GradCAM.

        Args:
            model: El modelo PyTorch entrenado.
            target_layer_name (str): Nombre de la capa objetivo para GradCAM.
            verbose (bool): Imprimir dimensiones de salida del mapa de saliencia.
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = dict()
        self.activations = dict()

        self._register_hooks()

        if verbose:
            print(f"GradCAM inicializado para la capa: {target_layer_name}")

    def _register_hooks(self):
        """
        Registra hooks para capturar activaciones y gradientes en la capa objetivo.
        """
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(self._save_activation)
                module.register_backward_hook(self._save_gradient)
                print(f"Hooks registrados en la capa: {name}")
                return
        raise ValueError(f"Capa {self.target_layer_name} no encontrada en el modelo.")

    def _save_activation(self, module, input, output):
        self.activations['value'] = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients['value'] = grad_output[0]

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Calcula el mapa GradCAM.

        Args:
            input (torch.Tensor): Tensor de entrada de forma (B, C, H, W).
            class_idx (int, opcional): Índice de la clase para el GradCAM.
                                       Si no se especifica, se usa la clase predicha.
            retain_graph (bool): Retener el grafo computacional (útil para GradCAM++ o backprop).

        Returns:
            saliency_map (torch.Tensor): Mapa de saliencia GradCAM.
            logit (torch.Tensor): Salida del modelo.
        """
        b, c, h, w = input.size()

        # Pasar por el modelo
        logit = self.model(input)

        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        if logit.dim() == 1 or logit.size(1) == 1:

            if class_idx is None:
                score = logit.squeeze()
            else:
                score = logit[:, class_idx].squeeze()
        else:
            print("MULTICLASE")
            if class_idx is None:
                score = logit[torch.arange(b), logit.argmax(dim=1)]
            else:
                score = logit[:, class_idx]

            score = score.sum() / b

        print("clase", score)

        self.model.zero_grad()
        print("Score requiere gradiente:", score.requires_grad)
        if not score.requires_grad:
            raise ValueError("El puntaje no requiere gradiente. Verifica tu entrada y modelo.")
        score.backward(retain_graph=retain_graph)

        gradients = self.gradients['value']
        print("gradientes", gradients)
        activations = self.activations['value']
        print("activaciones", activations)
        b, k, u, v = gradients.size()

        # Calcular pesos
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        print(weights)
        # Mapa de saliencia
        saliency_map = (weights * activations).sum(1, keepdim=True)
        print(saliency_map)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

        # Normalizar
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        """
        Llama al método `forward`.
        """
        return self.forward(input, class_idx, retain_graph)


def process_and_overlay_heatmaps(test_loader, test_loader_inter, gradcam, output_dir):
    """
    Genera mapas de GradCAM para test_loader y los solapa con las imágenes de test_loader_inter.
    Args:
        test_loader: DataLoader con las imágenes originales.
        test_loader_inter: DataLoader con las imágenes interpretables.
        gradcam: Objeto GradCAM inicializado.
        output_dir: Directorio base donde se guardarán las imágenes combinadas.
    """
    os.makedirs(output_dir, exist_ok=True)
        #
        # for (filenames,labels), (filenames_inter, _) in zip(test_loader, test_loader_inter):
        #     # Enviar imágenes al dispositivo
        #     images = cv2.imread(filenames)
        #     images_inter=  cv2.imread(filenames_inter)
        #

    for (images, labels, filenames), (images_inter, _, _) in zip(test_loader, test_loader_inter):
        # Enviar imágenes al dispositivo
        images = images.to(device)
        images_inter = images_inter.to(device)

        for i in range(len(images)):
            img = images[i].unsqueeze(0)
            img_inter = images_inter[i]
            filename = filenames[i]

            # Generar mapa GradCAM
            mask, logit = gradcam(img)
            heatmap, cam_result = visualize_cam(mask, img)

            resized_cam = F.interpolate(heatmap.unsqueeze(0), size=img_inter.shape[1:], mode='bilinear',
                                        align_corners=False)
            resized_cam = resized_cam.squeeze(0)  # Quitar dimensión extra

            combined_result = (resized_cam + img_inter.cpu()).div(2).clamp(0, 1)

            heatmap_path = modify_and_create_path(
                filename, old_folder="tab_to_image", new_folder="heatmaps"
            ).replace(".png", "_heatmap.png")

            combined_path = modify_and_create_path(
                filename, old_folder="tab_to_image", new_folder="heatmaps"
            ).replace(".png", "_combined.png")

            inter_img_path = modify_and_create_path(
                filename, old_folder="tab_to_image", new_folder="heatmaps"
            ).replace(".png", "_interpreted.png")

            # Guardar resultados
            save_image(resized_cam, heatmap_path)
            save_image(combined_result, combined_path)
            save_image(img_inter, inter_img_path)

            print(f"Guardado: {combined_path}")

def parse_arguments(parser):
    parser.add_argument('--dataset', default='fram', type=str)
    parser.add_argument('--noise_type', default='homogeneous', type=str)
    parser.add_argument('--augmented', default=False, type=bool)
    parser.add_argument('--channels', default=1, type=int)
    parser.add_argument('--seed', default=24, type=int)
    parser.add_argument('--n_jobs', default=14, type=int)

    return parser.parse_args()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='interpretability')
    args = parse_arguments(parser)

    n_procs = cpu_count()
    n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
    logger.info('n_jobs for tab_to_image_fs: {}'.format(n_jobs))

    if not args.augmented:
        path = os.path.join(cons.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.dataset}')
        path_images_interpretability = create_datasets(path=path, dataset_name=args.dataset, noise_type=args.noise_type,
                                                       interpretability=True)
        path_images = create_datasets(path=path, dataset_name=args.dataset, noise_type=args.noise_type,
                                      interpretability=False)
    else:
        test_path = os.path.join(cons.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.dataset}', 'ctgan',
                        'test_{}_seed_{}'.format(args.dataset, args.seed))
        train_path = os.path.join(cons.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.dataset}',
                                'ctgan',
                                'train_{}_seed_{}'.format(args.dataset, args.seed))


    if 'second' in args.dataset:
        model_file_path = str(os.path.join(cons.PATH_PROJECT_SAVE_MODEL,
                                           'model_cnn_classification_{}_{}.pt'.format(
                                               args.noise_type, 'fram')))
    else:
        model_file_path = str(os.path.join(cons.PATH_PROJECT_SAVE_MODEL,
                                           'model_cnn_classification_{}_{}.pt'.format(
                                               args.noise_type, args.dataset)))
    model = torch.load(model_file_path, map_location=torch.device('cpu'))

    split_sed_values = model['split_seed']
    if args.seed in split_sed_values:
        index = split_sed_values.index(args.seed)
    else:
       raise ValueError(f"Bad Seed: {args.seed} not valid.")


    trained_model = model['trained_model'][index]
    best_config = model['best_config'][index]


    device = torch.device('cpu')


    shape, train_subset, val_subset, test_subset, num_classes = cnn.dataset_partition(
                        dataset=args.dataset, noise_type=args.noise_type, channels=args.channels, min_shape=best_config['min_shape'], path_images=path_images, split_seed=args.seed, interpretability=False, gradcam=True)

    shape_inter, train_subset_inter, val_subset_inter, test_subset_inter, num_classes = cnn.dataset_partition(
                        dataset=args.dataset, noise_type=args.noise_type, channels=3, min_shape=best_config['min_shape'], path_images=path_images_interpretability, split_seed=args.seed, interpretability=True , gradcam=True
                    )
    print(shape)

    batch_size = int(len(train_subset)/best_config['n_batches']) + 1
    print(batch_size)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader_inter = DataLoader(test_subset_inter, batch_size=batch_size, shuffle=False, pin_memory=False)

    print(test_loader)
    print(len(test_loader))

    gradcam = GradCAM(trained_model, target_layer_name="cnn2", verbose=True)

    output_directory = "./gradcam_results"
    process_and_overlay_heatmaps(test_loader, test_loader_inter, gradcam, output_directory)
