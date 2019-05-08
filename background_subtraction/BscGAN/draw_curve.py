import matplotlib.pyplot as plt
import numpy as np


#display Loss G curve on training dataset

y = np.loadtxt('curves/array_err_g.out')
x = np.arange(1,100+1)
plt.plot(x, y, linewidth=2.0)

plt.xlabel('epoch')
plt.ylabel('loss G')
plt.title('Trainging curve')
plt.grid(True)
plt.savefig("curves/curve_Loss_G.png")
plt.show()


#display psnr curve on testing dataset

y = np.loadtxt('curves/array_psnr.out')
x = np.arange(1,100+1)
plt.plot(x, y, linewidth=2.0)

plt.xlabel('epoch')
plt.ylabel('psnr')
plt.title('Trainging curve')
plt.grid(True)
plt.savefig("curves/curve_psnr.png")
plt.show()

