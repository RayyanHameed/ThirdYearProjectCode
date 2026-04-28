import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from Tutorial_Codes import psnr, reproject, dxm, dym, dxp, dyp, function_TGV_denoising_CP, P_a_Huber, function_HuberTV_denoising_CP, function_TV_denoising_CP

#obtaining image and aritifically corrupting the image 
import imageio.v2 as imageio
import numpy as np
from skimage.transform import resize

def load_image_256(path):
    img = imageio.imread(path)

    # Convert to grayscale if RGB
    if img.ndim == 3:
        img = img[..., :3]
        img = img @ np.array([0.2989, 0.5870, 0.1140])

    img = img.astype(np.float64)

    # Normalize
    if img.max() > 1.5:
        img /= 255.0

    # Resize to 256×256 
    if img.shape != (256, 256):
        img = resize(img, (256, 256), anti_aliasing=True)

    return img


def corrupt_image(u, sigma, seed=0):
    random = np.random.default_rng(seed)
    f = u + sigma * random.standard_normal(u.shape)
    return np.clip(f, 0, 1)



def psnr(u1,u2):

    (n,m)=u1.shape
    mse=(1/(n*m))*(np.sum((u1-u2)**2))
    mpsnr=10*np.log10(1/mse)

    return mpsnr


def MSE(u_alpha,u_clean):
    return np.mean((u_alpha - u_clean)**2)


def golden_section_search(f, a, b, tol=1e-4):
    golden_ratio = (np.sqrt(5) - 1) / 2
    
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)
    
    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        
        c = b - golden_ratio * (b - a)
        d = a + golden_ratio * (b - a)
    
    return (a + b) / 2



def R2R_loss(y,alpha):
    (n,m)=y.shape

    N = n*m

    beta = 0.5

    sigma = 0.05 

    tau = (beta/(1-beta))**0.5

    
    loss = 0

    K = 7

    for k in range(1,K + 1):
        omega = sigma * np.random.standard_normal(y.shape)
        y1 = y + tau*omega
        y2 = y - (omega/tau)
        loss += (np.mean((function_TV_denoising_CP(y1,alpha,1000) - y2)**2))
    
    loss = 1/K * loss 

    return loss




random_k = np.random.default_rng()

def sub_sampler(image,random):
    rows = image.shape[0]
    cols = image.shape[1]
    
    g1 = np.zeros((rows//2, cols//2))
    g2 = np.zeros((rows//2, cols//2))

    for i in range(0, rows, 2):
        for j in range(0,cols,2):
            block = image[i:i+2,j:j+2]
            # print(block)

            neighbour = random.integers(1,5)

            if neighbour == 1:
                value1 = block[0][0]
                value2 = block[0][1]
            elif neighbour == 2:
                value1 = block[1][0]
                value2 = block[1][1]
            elif neighbour == 3:
                value1 = block[0][0]
                value2 = block[1][0]
            else:
                value1 = block[0][1]
                value2 = block[1][1]

            
            g1[i//2,j//2] = value1
            g2[i//2,j//2] = value2

    return g1,g2




def neighbour2neighbour_loss(image, alph):
    K = 7
    loss = 0
    f_of_y = function_TV_denoising_CP(image,alph,1000)
    for i in range(1,K+1):
        g1,g2 = sub_sampler(image, random_k)
        f_of_g1 = function_TV_denoising_CP(g1,alph,1000)
        
        l_rec = np.sum((f_of_g1 - g2)**2)
        g1_of_f, g2_of_f = sub_sampler(f_of_y, random_k)
        l_reg = np.sum((f_of_g1 - g2 - g1_of_f + g2_of_f)**2)
        
        loss = loss + l_rec + 0.5*l_reg
    

    return loss * 1/K




def plot_psnr(alphas, psnrs, selected_alphas=None, selected_psnrs=None):
    plt.figure(figsize=(6,4))
    plt.semilogx(alphas, psnrs)
    
    colors = {
        'SURE': 'blue',
        'Supervised': 'orange',
        'R2R': 'green',
        'N2N': 'red'
    }
    sizes = {
        'SURE': 200,
        'Supervised': 150,
        'R2R': 150,
        'N2N': 150
    }
    
    if selected_alphas is not None and selected_psnrs is not None:
        for name, alpha in selected_alphas.items():
            exact_psnr = selected_psnrs[name]
            plt.scatter(alpha, exact_psnr, label=name,
                       color=colors[name], zorder=5,
                       s=sizes[name],
                       edgecolors='black', linewidths=1)
    
    plt.xlabel("alpha")
    plt.ylabel("PSNR (dB)")
    plt.title("Alpha vs PSNR")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_psnr_linear(alphas, psnrs, selected_alphas=None):
    plt.figure(figsize=(8,5))
    plt.plot(alphas, psnrs)
    
    colors = {
        'SURE': 'blue',
        'Supervised': 'orange',
        'R2R': 'green',
        'N2N': 'red'
    }
    
    if selected_alphas is not None:
        for name, alpha in selected_alphas.items():
            idx = np.argmin(np.abs(alphas - alpha))
            plt.scatter(alpha, psnrs[idx], label=name,
                       color=colors[name], zorder=5, s=100)
    
    plt.xlabel("alpha")
    plt.ylabel("PSNR (dB)")
    plt.title("Alpha vs PSNR (linear scale)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 0.2)  # zoom in on relevant region
    plt.show()


def plot_mse(alphas, mses):
    plt.figure(figsize=(6,4))
    plt.plot(alphas, mses)
    plt.xlabel("alpha")
    plt.ylabel("MSE")
    plt.title("Alpha vs MSE")
    plt.grid(True)
    plt.show()



def find_optimal_alpha(noisy_img, clean_img):
    result = golden_section_search(
        lambda a: MSE(function_TV_denoising_CP(noisy_img, a, 1000), clean_img),
        a=0.001, b=1.0
    )
    # still need psnr list for plotting
    alphas = np.linspace(0.01, 1.0, 100)
    psnrs = [psnr(function_TV_denoising_CP(noisy_img, a, 1000), clean_img) for a in alphas]
    return result, psnrs


def find_optimal_alpha_R2R_loss(noisy_img):
    return golden_section_search(
        lambda a: R2R_loss(noisy_img, a),
        a=0.001, b=1.0
    )

def find_optimal_alpha_N2N_loss(noisy_img):
    return golden_section_search(
        lambda a: neighbour2neighbour_loss(noisy_img, a),
        a=0.001, b=1.0
    )
    

def SURE_loss(image,alph):
    y = image 
    n = y.size
    omega = np.random.standard_normal(y.shape)
    f_of_y = function_TV_denoising_CP(y,alph,1000)
    sigma = 0.05 
    tau = 0.01*sigma#paper recommends choosing tau to be 1% of standard deviation
    delta = tau*omega 
    f_of_y_plus_delta = function_TV_denoising_CP((y+delta),alph,1000)
    component_1 = (1/n) * np.sum((f_of_y - y)**2)
    component_2 = (2*(sigma**2)/(n*tau)) * np.dot(omega.flatten(),(f_of_y_plus_delta - f_of_y).flatten())
    loss = component_1 + component_2 - sigma**2
    return loss 

def find_optimal_alpha_SURE_loss(noisy_img):
    return golden_section_search(
        lambda a: SURE_loss(noisy_img, a),
        a=0.001, b=1.0
    )


def break_image(image, n):
    size_of_image = image.shape
    rows = size_of_image[0]
    cols = size_of_image[1]
    image_blocks = []
    
    for i in range(0, rows, n):
        for j in range(0,cols,n):
            block = image[i:i+n,j:j+n]
            image_blocks.append(block)
    

    return image_blocks


def find_spatially_varying_alpha_supervised(image_blocks, clean_test_image):
    spatially_varying_alpha = []
    reconstructed_blocks = []
    broken_clean_images = break_image(clean_test_image, 16)
    alphas_grid = np.linspace(0.001, 1.0, 100)
    
    for i in range(len(image_blocks)):
        print(i)
        mses = [MSE(function_TV_denoising_CP(image_blocks[i], a, 1000), broken_clean_images[i]) 
                for a in alphas_grid]
        best_alpha = alphas_grid[np.argmin(mses)]
        print(f"Block {i} best alpha: {best_alpha:.4f}")
        spatially_varying_alpha.append(best_alpha)
        reconstructed_blocks.append(function_TV_denoising_CP(image_blocks[i], best_alpha, 1000))
    
    
    return reconstructed_blocks, spatially_varying_alpha

def rebuild_image(reconstructed_blocks, n):
    rebuilt_image = np.zeros((256, 256))
    block_index = 0
    for i in range(0, 256, n):
        for j in range(0, 256, n):
            rebuilt_image[i:i+n, j:j+n] = reconstructed_blocks[block_index]
            block_index += 1
    return rebuilt_image


def find_spatially_varying_alpha_SURE(image_blocks):
    spatially_varying_alpha = []
    reconstructed_blocks = []
    alphas_grid = np.linspace(0.001, 0.2, 100)
    
    for i in range(len(image_blocks)):
        print(i)
        SURE_losses = [SURE_loss(image_blocks[i], a) for a in alphas_grid]
        best_alpha = alphas_grid[np.argmin(SURE_losses)]
        print(f"Block {i} best alpha: {best_alpha:.4f}")
        spatially_varying_alpha.append(best_alpha)
        reconstructed_blocks.append(function_TV_denoising_CP(image_blocks[i], best_alpha, 1000))
    
    return reconstructed_blocks, spatially_varying_alpha






