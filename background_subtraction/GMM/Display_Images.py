import matplotlib.pyplot as plt

def Display_2_Images(ima1, ima2, name1, name2):
    f, axarr = plt.subplots(2, 1)
    # plt.suptitle(suptitle)
    axarr[0].imshow(ima1, cmap='Greys_r')
    axarr[0].title.set_text(name1)
    axarr[1].imshow(ima2, cmap='Greys_r')
    axarr[1].title.set_text(name2)
    plt.show()


def Display_4_Images(ima1, ima2, ima3, ima4, name1, name2, name3, name4):
    f, axarr = plt.subplots(2, 2)
    # plt.suptitle(suptitle)
    axarr[0, 0].imshow(ima1, cmap='Greys_r')
    axarr[0, 0].title.set_text(name1)
    axarr[0, 1].imshow(ima2, cmap='Greys_r')
    axarr[0, 1].title.set_text(name2)
    axarr[1, 0].imshow(ima3, cmap='Greys_r')
    axarr[1, 0].title.set_text(name3)
    axarr[1, 1].imshow(ima4, cmap='Greys_r')
    axarr[1, 1].title.set_text(name4)
    plt.show()


def Display_6_Images(ima1, ima2, ima3, ima4, ima5, ima6, name1, name2, name3, name4, name5, name6):
    f, axarr = plt.subplots(3, 2)
    # plt.suptitle(suptitle)
    axarr[0, 0].imshow(ima1, cmap='Greys_r')
    axarr[0, 0].title.set_text(name1)
    axarr[0, 1].imshow(ima2, cmap='Greys_r')
    axarr[0, 1].title.set_text(name2)
    axarr[1, 0].imshow(ima3, cmap='Greys_r')
    axarr[1, 0].title.set_text(name3)
    axarr[1, 1].imshow(ima4, cmap='Greys_r')
    axarr[1, 1].title.set_text(name4)
    axarr[2, 0].imshow(ima5, cmap='Greys_r')
    axarr[2, 0].title.set_text(name5)
    axarr[2, 1].imshow(ima6, cmap='Greys_r')
    axarr[2, 1].title.set_text(name6)
    plt.show()