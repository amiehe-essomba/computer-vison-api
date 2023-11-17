import cv2
import imageio
import tempfile
import shutil

# Fonction pour capturer des images depuis la webcam et les afficher dans une fenêtre
def show_webcam(video_name="output_video.mp4", format="mp4", fps=30):
    fmt = {"mp4": 'mp4v', "avi": "XVID"}
    temp_video_dir = tempfile.TemporaryDirectory(suffix=".mp4")

    if format in fmt.keys():
        if fps > 0:
            print("Press 'r' to start recording, 's' to stop recording, and 'q' to exit.")

            video_capture = cv2.VideoCapture(0)  # Utilisez 0 pour la webcam intégrée, ajustez si nécessaire
            recording = False
            video_writer = None

            try:
                while True:
                    _, frame = video_capture.read()
                    cv2.imshow('Webcam', frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('r'):
                        if not recording:
                            print('Recording started...')
                            recording = True
                            fourcc = cv2.VideoWriter_fourcc(*fmt['mp4'])
                            video_writer = cv2.VideoWriter(
                                f"{temp_video_dir.name}/temp_video.mp4", fourcc, fps, (608, 608), isColor=True)
                            
                    elif key == ord('s'):
                        if recording:
                            print('Recording stopped...')
                            recording = False
                            video_writer.release()
                            shutil.move(f"{temp_video_dir.name}/temp_video.mp4", video_name)

                    elif key == ord('q'):
                        break

                    if recording:
                        video_writer.write(frame)

            except KeyboardInterrupt:
                pass

            finally:
                video_capture.release()
                cv2.destroyAllWindows()

                with open(f'{temp_video_dir.name}/temp_video.mp4', 'rb') as int_file:
                    # Lire le contenu du fichier temporaire
                    video_data = int_file.read()

                with open('_.mp4', 'wb') as out_file:
                    # Lire le contenu du fichier temporaire
                    out_file.write(video_data)
                
                shutil.rmtree(temp_video_dir.name, ignore_errors=True)

    else:
        print(f"{format} not in {list(fmt.keys())}")





