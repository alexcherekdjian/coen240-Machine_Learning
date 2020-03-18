from PIL import Image
from math import sqrt, pi, exp

# to use code for portrait or family, change filename 
filename = "family"

# open both real image and ground truth
im = Image.open(filename + ".jpg")
im_gnd = Image.open(filename + ".png")

# load pixels for real image and ground truth
px = im.load()
px_gnd = im_gnd.load()

# pixels are skin constants ie.gnd=white
u_1r = 0
u_1g = 0
sigma_1r2 = 0
sigma_1g2 = 0
sigma_1r = 0
sigma_1g = 0
N_1 = 0

# pixels are background constants ie.gnd=black
u_0r = 0
u_0g = 0
sigma_0r2 = 0
sigma_0g2 = 0
sigma_0r = 0
sigma_0g = 0
N_0 = 0

# calculate u's and N's
for x in range(0, im.size[0]):
    for y in range(0, im.size[1]):
        
        # get pixel values of real image            
        r_i = px[x,y][0]
        g_i = px[x,y][1] 
                
        # get pixel values of ground truth
        r_i_gnd = px_gnd[x,y][0]
        g_i_gnd = px_gnd[x,y][1]      
        
        if r_i_gnd == 0 and g_i_gnd == 0:
            # pixels are background
            u_0r += r_i
            u_0g += g_i   
            N_0 += 1
            
        else:
            # pixels are skin
            u_1r += r_i
            u_1g += g_i
            N_1 += 1

u_0r /= N_0
u_0g /= N_0
u_1r /= N_1
u_1g /= N_1

true_values_dict = {}

# calculate sigmas and sigmas^2
for x in range(0, im.size[0]):
    for y in range(0, im.size[1]):
        
        # get pixel values of real image
        r_i = px[x,y][0]
        g_i = px[x,y][1] 
    
        # get pixel values of ground truth
        r_i_gnd = px_gnd[x,y][0]
        g_i_gnd = px_gnd[x,y][1]      
            
        if r_i_gnd == 0 and g_i_gnd == 0:
            # pixels are background
            sigma_0r2 += (r_i - u_0r)**2
            sigma_0g2 += (g_i - u_0g)**2
            true_values_dict[x,y] = "background"
  
        else:
            # pixels are skin
            sigma_1r2 += (r_i - u_1r)**2
            sigma_1g2 += (g_i - u_1g)**2
            true_values_dict[x,y] = "skin"

sigma_0r2 /= N_0
sigma_0g2 /= N_0
sigma_1r2 /= N_1
sigma_1g2 /= N_1    

sigma_0r = sqrt(sigma_0r2)
sigma_0g = sqrt(sigma_0g2)
sigma_1r = sqrt(sigma_1r2)
sigma_1g = sqrt(sigma_1g2)

# create new image for binary mask
im_mask = Image.new('RGB', (im.size[0], im.size[1]), color = 'black')
px_mask = im_mask.load()

predicted_values_dict = {}

# calculate probabilities
for x in range(0, im.size[0]):
    for y in range(0, im.size[1]):
        
        # get pixel values of real image
        r_k = px[x,y][0]
        g_k = px[x,y][1]
        
        # calculate first constants
        c0_r = 1/(sqrt(2*pi)*sigma_0r)
        c0_g = 1/(sqrt(2*pi)*sigma_0g)
        c1_r = 1/(sqrt(2*pi)*sigma_1r)
        c1_g = 1/(sqrt(2*pi)*sigma_1g)
        
        # calculate probabilities
        px_h0 = c0_r * exp(-0.5 * (((r_k - u_0r)**2)/sigma_0r2)) * c0_g * exp(-0.5 * (((g_k - u_0g)**2)/sigma_0g2))
        px_h1 = c1_r * exp(-0.5 * (((r_k - u_1r)**2)/sigma_1r2)) * c1_g * exp(-0.5 * (((g_k - u_1g)**2)/sigma_1g2))
        
        # if the pixel is skin, color pixel in mask white
        if (px_h1/px_h0) > (px_h0/px_h1): 
            px_mask[x,y] = (255,255,255)
            predicted_values_dict[x,y] = "skin"
        else:
        	predicted_values_dict[x,y] = "background"

# tp variables
true_skin = 0
predicted_skin = 0

# tn variables
true_background = 0
predicted_background = 0

# fp variables
predicted_skin_true_background = 0

# fp variables
predicted_background_true_skin = 0

# calculate rates
tp_rate = 0
tn_rate = 0
fp_rate = 0
fn_rate = 0

# getting other values
for x in range(0, im.size[0]):
	for y in range(0, im.size[1]):

		# get pixel values of real image
		if predicted_values_dict[x,y] == "background":
			# background else skin
			true_background += 1
		else:
			true_skin += 1

		if (true_values_dict[x,y] == "skin") and (predicted_values_dict[x,y] == "skin"):
			# for tp
			predicted_skin += 1

		elif(true_values_dict[x,y] == "background") and (predicted_values_dict[x,y] == "background"):
			# for tn
			predicted_background += 1

		elif(true_values_dict[x,y] == "background") and (predicted_values_dict[x,y] == "skin"):
			# for fp
			predicted_skin_true_background += 1

		elif(true_values_dict[x,y] == "skin") and (predicted_values_dict[x,y] == "background"):
			# for fn
			predicted_background_true_skin +=1

tp_rate = predicted_skin / true_skin * 100
tn_rate = predicted_background / true_background * 100
fp_rate = predicted_skin_true_background / true_background * 100
fn_rate = predicted_background_true_skin /true_skin * 100

print("true positive rate = " + str(round(tp_rate, 2)))
print("true negative rate = " + str(round(tn_rate, 2)))
print("false positive rate = " + str(round(fp_rate, 2)))
print("false negative rate = " + str(round(fn_rate, 2)))

# save the final binary mask
im_mask.save(filename + '_mask.png')
