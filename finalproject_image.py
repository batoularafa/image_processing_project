import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Path to the directory containing medical images
image_dir = 'C:\\Users\\batou\\Downloads\\mri'
# Function to load images
def load_images(image_dir):
    images = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
            img = cv2.imread(os.path.join(image_dir, file_name), cv2.IMREAD_GRAYSCALE)
            images.append(img)
    return images

def histogram_equalization(image):
    #gray_sc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(image)
def contrast_stretching(image, alpha=5, beta=50):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def canny_edge_detector(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def high_pass_filter(image):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)



def calculate_contrast(image):
    return image.std()

def contrast_enhancement_factor(original, enhanced):
    return calculate_contrast(enhanced) / calculate_contrast(original)

def edge_detection_metrics(original_edges, detected_edges):
    precision = precision_score(original_edges.flatten(), detected_edges.flatten(),pos_label=0, zero_division=0)
    recall = recall_score(original_edges.flatten(), detected_edges.flatten(),pos_label=0, zero_division=0)
    f1 = f1_score(original_edges.flatten(), detected_edges.flatten(), pos_label=0, zero_division=0)
    return precision, recall, f1

def close_all_figures(event):
    if event.key == '0':
        plt.close('all')

def plotting(img1, img2, img3, img4, img5):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,3,1)
    ax1.imshow(img1, cmap='gray')
    ax1.set_title('original')
    ax1.axis('off')
    ax2 = fig.add_subplot(2,3,2)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(f'Histogram Equalization\nContrast Factor: {ce_factor_he:.2f}')
    ax2.axis('off')
    ax3 = fig.add_subplot(2,3,3)
    ax3.imshow(img3, cmap='gray')
    ax3.set_title(f'Contrast Stretching\nContrast Factor: {ce_factor_cs:.2f}')
    ax3.axis('off')
    ax4 = fig.add_subplot(2,3,4)
    ax4.imshow(img4, cmap='gray')
    ax4.set_title("canny edge detector")
    ax4.axis('off')
    ax5 = fig.add_subplot(2,3,5)
    ax5.imshow(img5, cmap='gray')
    ax5.set_title("high pass filter")
    ax5.axis('off')
    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', close_all_figures)
    return plt.show()


if __name__== "__main__":
    ce_factors_he = []
    ce_factors_cs = []
    precisions_canny = []
    recalls_canny = []
    f1s_canny = []
    precisions_hp = []
    recalls_hp = []
    f1s_hp = []
    
    img = load_images(image_dir)
    print(len(img))
    for i in img:
        h_eq = histogram_equalization(i)
        cs_image = contrast_stretching(i)
        canny_edges = canny_edge_detector(i)
        hp_edges = high_pass_filter(i)
        # h_eq1 = np.array(h_eq)
        # cs_image1= np.array(cs_image)
        # canny_edges1 = np.array(canny_edges)
        # hp_edges1 = np.array(hp_edges)

        ce_factor_he = contrast_enhancement_factor(i, h_eq)
        ce_factor_cs = contrast_enhancement_factor(i, cs_image)
        precision_canny, recall_canny, f1_canny = edge_detection_metrics(canny_edges, canny_edges)  # Ideally compare to ground truth
        precision_hp, recall_hp, f1_hp = edge_detection_metrics(hp_edges, hp_edges)  # Ideally compare to ground truth
        
        # Append metrics to lists
        ce_factors_he.append(ce_factor_he)
        ce_factors_cs.append(ce_factor_cs)
        precisions_canny.append(precision_canny)
        recalls_canny.append(recall_canny)
        f1s_canny.append(f1_canny)
        precisions_hp.append(precision_hp)
        recalls_hp.append(recall_hp)
        f1s_hp.append(f1_hp)

        plotting(i,h_eq, cs_image, canny_edges, hp_edges)

        # fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        # axs[0, 0].imshow(i, cmap='gray')
        # axs[0, 0].set_title('Original Image')
        # axs[0, 1].imshow(h_eq, cmap='gray')
        # axs[0, 1].set_title(f'Histogram Equalization\nContrast Factor: {ce_factor_he:.2f}')
        # axs[0, 2].imshow(cs_image, cmap='gray')
        # axs[0, 2].set_title(f'Contrast Stretching\nContrast Factor: {ce_factor_cs:.2f}')
        # axs[1, 0].imshow(canny_edges, cmap='gray')
        # axs[1, 0].set_title(f'Canny Edge Detection\nPrecision: {precision_canny:.2f}, Recall: {recall_canny:.2f}, F1: {f1_canny:.2f}')
        # axs[1, 1].imshow(hp_edges, cmap='gray')
        # axs[1, 1].set_title(f'High-Pass Filter\nPrecision: {precision_hp:.2f}, Recall: {recall_hp:.2f}, F1: {f1_hp:.2f}')
        # axs[1, 2].axis('off')
        # plt.tight_layout()
        # plt.show()

    print(f'Average Contrast Enhancement Factor (Histogram Equalization): {np.mean(ce_factors_he):.2f}')
    print(f'Average Contrast Enhancement Factor (Contrast Stretching): {np.mean(ce_factors_cs):.2f}')
    print(f'Average Precision (Canny Edge Detection): {np.mean(precisions_canny):.2f}')
    print(f'Average Recall (Canny Edge Detection): {np.mean(recalls_canny):.2f}')
    print(f'Average F1 Score (Canny Edge Detection): {np.mean(f1s_canny):.2f}')
    print(f'Average Precision (High-Pass Filter): {np.mean(precisions_hp):.2f}')
    print(f'Average Recall (High-Pass Filter): {np.mean(recalls_hp):.2f}')
    print(f'Average F1 Score (High-Pass Filter): {np.mean(f1s_hp):.2f}')

        # plotting(i,h_eq)
        # cv2.imshow('image',i)
        # cv2.imshow('h_eq', h_eq)
        # cv2.imshow('contrast', cs_image)
        # cv2.imshow('canny edge', canny_edges)
        # cv2.imshow('high pass', hp_edges)
        # cv2.waitKey(0)

