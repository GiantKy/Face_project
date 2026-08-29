import cv2


def draw_pose_info(frame, pose, text, valid):

    color = (0, 255, 0) if valid else (0, 0, 255)

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if pose is None:
        return frame

    yaw = pose["yaw"]
    pitch = pose["pitch"]
    roll = pose["roll"]

    cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, f"Roll: {roll:.2f}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame