
import time
from CameraPoses import *

from matplotlib import pyplot as plt

import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.rotations as pr
import pytransform3d.camera as pc

from cycler import cycle

with open('intrinsicNew.npy', 'rb') as f:
    intrinsic = np.load(f)

skip_frames = 2
data_dir = ''
vo = CameraPoses(data_dir, skip_frames, intrinsic)

gt_path = []
estimated_path = []
camera_pose_list = []
start_pose = np.ones((3, 4))
start_translation = np.zeros((3, 1))
start_rotation = np.identity(3)
start_pose = np.concatenate((start_rotation, start_translation), axis=1)
# print("Start pose: ", start_pose)

cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream")

process_frames = False
old_frame = None
new_frame = None
frame_counter = 0

cur_pose = start_pose

while (cap.isOpened()):

    # Capture frame-by-frame
    ret, new_frame = cap.read()

    frame_counter += 1

    start = time.perf_counter()

    if process_frames and ret:
        q1, q2 = vo.get_matches(old_frame, new_frame)
        if q1 is not None:
            if len(q1) > 20 and len(q2) > 20:
                transf = vo.get_pose(q1, q2)
                cur_pose = cur_pose @ transf

        hom_array = np.array([[0, 0, 0, 1]])
        hom_camera_pose = np.concatenate((cur_pose, hom_array), axis=0)
        camera_pose_list.append(hom_camera_pose)
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

        estimated_camera_pose_x, estimated_camera_pose_y = cur_pose[0, 3], cur_pose[2, 3]

    elif process_frames and ret is False:
        break

    old_frame = new_frame

    process_frames = True

    end = time.perf_counter()

    total_time = end - start
    fps = 1 / total_time

    cv2.putText(new_frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(new_frame, str(np.round(cur_pose[0, 0], 2)), (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[0, 1], 2)), (340, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[0, 2], 2)), (420, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[1, 0], 2)), (260, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[1, 1], 2)), (340, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[1, 2], 2)), (420, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[2, 0], 2)), (260, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[2, 1], 2)), (340, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[2, 2], 2)), (420, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

    cv2.putText(new_frame, str(np.round(cur_pose[0, 3], 2)), (540, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[1, 3], 2)), (540, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[2, 3], 2)), (540, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow("img", new_frame)

    # if frame_counter % 20 == 0:
    # print("FPS: ", fps)
    # print("Frames: ", frame_counter)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the video capture object
cap.release()

"""for i in tqdm(range(number_of_frames)):
    if i == 0:
        cur_pose = start_pose
    else:
        q1, q2 = vo.get_matches(i)
        if len(q1) > 20 and len(q2) > 20:
            transf = vo.get_pose(q1, q2)
            cur_pose = cur_pose @ transf

    hom_array = np.array([[0,0,0,1]])
    hom_camera_pose = np.concatenate((cur_pose,hom_array), axis=0)
    camera_pose_list.append(hom_camera_pose)
    estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

    estimated_camera_pose_x, estimated_camera_pose_y = cur_pose[0, 3], cur_pose[2, 3]
"""

number_of_frames = 20
image_size = np.array([640, 480])
# image_size = np.array([1920, 1080])

plt.figure()
# ax = pt.plot_transform()
ax = plt.axes(projection='3d')

camera_pose_poses = np.array(camera_pose_list)

key_frames_indices = np.linspace(0, len(camera_pose_poses) - 1, number_of_frames, dtype=int)
colors = cycle("rgb")

for i, c in zip(key_frames_indices, colors):
    pc.plot_camera(ax, vo.K, camera_pose_poses[i],
                   sensor_size=image_size, c=c)

plt.show()

take_every_th_camera_pose = 2

estimated_path = np.array(estimated_path[::take_every_th_camera_pose])

plt.plot(estimated_path[:, 0], estimated_path[:, 1])
plt.show()
