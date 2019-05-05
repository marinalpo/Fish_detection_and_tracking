import matplotlib.pyplot as plt


def Display_Image(ima, name):
    f, axarr = plt.subplots(1, 1)
    # plt.suptitle(suptitle)
    axarr.imshow(ima, cmap='Greys_r')
    axarr.title.set_text(name)
    plt.show()


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
    f, axarr = plt.subplots(2, 3)
    # plt.suptitle(suptitle)
    axarr[0, 0].imshow(ima1, cmap='Greys_r')
    axarr[0, 0].title.set_text(name1)
    axarr[0, 1].imshow(ima2, cmap='Greys_r')
    axarr[0, 1].title.set_text(name2)
    axarr[0, 2].imshow(ima3, cmap='Greys_r')
    axarr[0, 2].title.set_text(name3)
    axarr[1, 0].imshow(ima4, cmap='Greys_r')
    axarr[1, 0].title.set_text(name4)
    axarr[1, 1].imshow(ima5, cmap='Greys_r')
    axarr[1, 1].title.set_text(name5)
    axarr[1, 2].imshow(ima6, cmap='Greys_r')
    axarr[1, 2].title.set_text(name6)
    plt.show()

# f, axarr = plt.subplots(2, 2)
# axarr[0,0].imshow(ori1, cmap='Greys_r')
# axarr[0,0].title.set_text('Original Frame #145')
# rect1 = patches.Rectangle((boxes1[0,0], boxes1[0,1]),boxes1[0,2],boxes1[0,3],linewidth=2,edgecolor='g',facecolor='none')
# rect2 = patches.Rectangle((boxes1[1,0], boxes1[1,1]),boxes1[1,2],boxes1[1,3],linewidth=2,edgecolor='r',facecolor='none')
# axarr[0,0].add_patch(rect1)
# axarr[0,0].add_patch(rect2)
#
# axarr[0,1].imshow(hull1, cmap='Greys_r')
# axarr[0,1].scatter(centroids1[0,0], centroids1[0,1], color='g', label='Fish 1')
# axarr[0,1].scatter(centroids1[1,0], centroids1[1,1], color='r', label='Fish 2')
# axarr[0,1].title.set_text('Centroids Frame #145')
#
# axarr[1,0].imshow(ori2, cmap='Greys_r')
# axarr[1,0].title.set_text('Original Frame #160')
# rect1 = patches.Rectangle((boxes2[0,0], boxes2[0,1]),boxes2[0,2],boxes2[0,3],linewidth=2,edgecolor='g',facecolor='none')
# rect2 = patches.Rectangle((boxes2[1,0], boxes2[1,1]),boxes2[1,2],boxes2[1,3],linewidth=2,edgecolor='r',facecolor='none')
# axarr[1,0].add_patch(rect1)
# axarr[1,0].add_patch(rect2)
#
# axarr[1,1].imshow(hull2, cmap='Greys_r')
# axarr[1,1].scatter(centroids2[0,0], centroids2[0,1], color='g', label='Fish 1')
# axarr[1,1].scatter(centroids2[1,0], centroids2[1,1], color='r', label='Fish 2')
# axarr[1,1].title.set_text('Centroids Frame #160')
# plt.show()