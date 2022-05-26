import cv2

def show_mask():
    mask_path = '/data/szy4017/code/detr/mask.png'
    img_path = '/data/szy4017/code/detr/img.png'

    mask = cv2.imread(mask_path)
    img = cv2.imread(img_path)

    h, w, c = mask.shape
    H, W, C = img.shape
    mask = cv2.resize(mask, (W, H))

    maskaddimg = cv2.add(img, mask)
    cv2.imwrite('maskaddimg.png', maskaddimg)



if __name__ == '__main__':
    show_mask()