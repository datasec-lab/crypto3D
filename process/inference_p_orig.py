import torch
import numpy as np
import C3D_model
import cv2
import time

torch.backends.cudnn.benchmark = True

def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()

    model = C3D_model.C3D(num_classes=101)
    torch.save(model.state_dict(), 'c3d-pretrained.pth')


    checkpoint = torch.load('c3d-pretrained.pth', map_location=lambda storage, loc: storage)
    #model.load_state_dict(checkpoint['state_dict'])

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()


    # Specify the full path to your video file
    video = "/path/to/your/video.avi"
    cap = cv2.VideoCapture(video)
    retaining = True

    totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("The total number of frames in this video is ", totalframecount)

    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            with open("result.txt", "a") as outfile:
                outfile.write(str(class_names[label])+str(probs[0][label])+'\n')
            clip.pop(0)

if __name__ == '__main__':
    main()




