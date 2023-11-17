from demo import encapsulation as lf 

class demo(lf.Wrapper):
    df                  : dict = {'label' : [], 'score':[], 'top':[], "left":[], "bottom":[], 'right':[]}
    colors              : dict = None 
    tracker             : str  = None
    youtube             : bool = False
    PATH                : str  = None 
    response            : bool = False
    save_file           : str  = "video_data.mp4"
    duration_in_second  : float = None 
    video_start         : int = None
    video_step          : int = 1
    alpha               : int = 30
    mode                : str = 'gray'
    only_mask           : bool = False 
    with_names          : bool = True

    def __init__(self, model_name: str = 'yolov8n.pt') -> None:
        super().__init__(model_name)
        self.model_name = model_name
    def build_model(self, is_seg : bool = False, **kwargs):

        MODEL       = self.models(is_seg=is_seg)
        
        if MODEL is None: pass 
        else:
            self._mod_demo_ = model_demo(
                duration_in_second=self.duration_in_second, 
                video_start=self.video_start, video_step=self.video_step,
                youtube=self.youtube, df=self.df, response=self.response, colors=self.colors, model=MODEL,
                seg=is_seg
                )

            self.tracker = self.tracker_check()

            if self.tracker is not True:
                kwargs['alpha'] = self.alpha 
                kwargs['mode'] = self.mode 
                kwargs['only_mask'] = self.only_mask
                kwargs['with_names'] = self.with_names 

                if self.tracker is None:
                    
                    video_data = self._mod_demo_.sumple_model(path_or_url=self.PATH, 
                                         name=self.model_name, save=self.save_file, **kwargs)

                else:
                    if not is_seg:
                        if self.model_name == "yolov8n.pt":

                            video_data = self._mod_demo_.model_track(path_or_url=self.PATH, 
                                            tracker=self.tracker, save=self.save_file, **kwargs)
                            
                        else: print("Tracking only works with 'yolov8n.pt'")
                    else: print("You cannot use segmentation and tracking together")
            else: pass

    def get_vido_info(self):
        self._mod_demo_ = model_demo(youtube=self.youtube )   
        self._mod_demo_.read(path_or_url=self.PATH, youtube=self.youtube, key=False)
        
  
class video_params:
    def __init__(self,
                duration_in_second : float = None, 
                video_start  : int = None,
                video_step : int = 1
                ) -> None:
        
        self.duration_in_second = duration_in_second
        self.video_start        = video_start
        self.video_step         = video_step

    def transformation(self, *details) -> tuple[int|None, int|None, int|None, None|str]:
        [fps, video_frame, duration] = details 
        error, second, step, start, end = None, None, None, None, None 

        if self.duration_in_second and type(self.duration_in_second) == type(int()):
            if 0 < self.duration_in_second <= int(duration):
                second = int(self.duration_in_second)
                if second > duration: second -= 1
            else: error = f"<duration_in_second> sould be positive and lower than {int( duration) if int( duration) < duration else int( duration) -1 }"
        else:
            second = int(duration)
            if second > duration: second -= 1
        
        if error is None:
            if self.video_start:
                if 0 <= self.video_start < int(video_frame - fps * second):
                    start = self.video_start
                else: error = f"<video_start> sould be positive and lower than {video_frame - fps * second}"
            else:  start = 0 

            if error is None :
                if self.video_step:
                    if (0 < self.video_step < 10 ) :
                        step = self.video_step 
                    else:
                        error =  error = f"<video_step> sould be positive and lower than {10}"
                else:  step = 1

                if error is None:
                    end = int(start + fps * second)

        return start, end, step, error

    def read_local_link(self, path : str):
        from pytube import YouTube
        import cv2

        vid_cap = None
        vid_cap, fps, total_frames, duration_seconds  =None, None, None, None 
        if path:
            [fps, total_frames, duration_seconds] = [None, None, None]
            error = None 
            try:   
                if not error:
                    vid_cap     = cv2.VideoCapture(path)
                    total_frames        = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps                 = vid_cap.get(cv2.CAP_PROP_FPS)
                    duration_seconds    = total_frames / fps
                else: print(error)
            except Exception as e: 
                print('Cannot open this file : {e}')
        else:   print('invalid path')

        return vid_cap, fps, total_frames, duration_seconds

    def read_youtube_link(self, url : str = "") -> tuple[any, float, int, float]:
        from pytube import YouTube
        import cv2

        vid_cap = None
        vid_cap, fps, total_frames, duration_seconds  =None, None, None, None

        if url:
            [fps, total_frames, duration_seconds] = [None, None, None]

            youtube_link = "https://www.youtube.com"

            error = None 
            try:
                try:
                    if str(url).rstrip().lstrip()[:len(youtube_link)] == "https://www.youtube.com": pass 
                    else: error = "Your url is not a YouTube link."
                except IndexError:
                    error = "Your url is not a YouTube link."
                    
                if not error:
                    yt          = YouTube(url)
                    stream      = yt.streams.filter(file_extension="mp4", res=720).first()
                    vid_cap     = cv2.VideoCapture(stream.url)
                    total_frames        = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps                 = vid_cap.get(cv2.CAP_PROP_FPS)
                    duration_seconds    = total_frames / fps
                else: print(error)
            except Exception as e: 
                print('Cannot open this link : {e}')
        else: print('invalid url')

        return vid_cap, fps, total_frames, duration_seconds
    
class model_demo(video_params, lf.YOLO_MODEL):
    def __init__(self, 
        duration_in_second      : float = None, 
        video_start             : int = None, 
        video_step              : int = 1,
        youtube                 : bool = False, 
        df                      : dict = {}, 
        response                : bool = False, 
        colors                  : dict = {}, 
        model                   : any = None,
        seg                     : bool  = False 
        ) -> None:
        
        super().__init__(duration_in_second, video_start, video_step)

        self.youtube    = youtube
        self.df         = df 
        self.model      = model 
        self.colors     = colors 
        self.response   = response 
        self.seg        = seg

    def read(self, path_or_url : str, youtube : bool=False, key : bool = False):
        if youtube is True:
            video, *details         = self.read_youtube_link(url=path_or_url)
        else:
            video, *details         = self.read_local_link(path=path_or_url )

        if key is False:
            if video:
                print(f"video details : \n\n")
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print(f'frame per second : {details[0]}\nvideo frame : {details[1]}\nduration : {round(details[-1], 4)}')
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        else:

            return video, details
        
    def is_youtube(self, url : str):
        video, details         = self.read(path_or_url=url, youtube=True, key=True)
        start, end, step, error, fps = [None, None, None, None, None]
        
        if video:
            fps = details[0]
            start, end, step, error = self.transformation(*details)
            details = (start, end, step)

        return video, details, fps, error

    def isno_youtube(self, path : str):
        video, details         = self.read(path_or_url=path, youtube=False, key=True)
        start, end, step, error, fps = [None, None, None, None, None]
        
        if video:   
            fps = details[0]     
            start, end, step, error = self.transformation(*details)
   
            details = (start, end, step)
          
        return video, details,  fps, error

    def sumple_model(self, path_or_url, name : str, save: str, **kwargs):
        if self.youtube is True:
            video, details, fps, error = self.is_youtube(url=path_or_url)
        else:
            video, details, fps, error = self.isno_youtube(path=path_or_url)

        video_data = None 
        if error is None:
        
            kwargs["fps"] = fps

            if name in ['yolov8n.pt', 'yolov8n-seg.pt']:
                if self.seg is False:
                    video_data = self.yolovo_video_demo(video=video, df=self.df, 
                        details=details, response=self.response, colors=self.colors, model=self.model, save=save, **kwargs)
                elif self.seg is True:
                    video_data = self.yolovo_video_seg_demo(video=video, df=self.df, 
                        details=details, response=self.response, colors=self.colors, model=self.model, save=save, **kwargs)
                else:
                    print("is_seg is not a bool type")
            else:
                video_data = self.my_model(video=video, df=self.df, 
                        details=details, response=self.response, colors=self.colors, model=self.model, save=save, **kwargs)
        else:  print(error)

        return video_data
    
    def model_track(self, path_or_url, tracker, save, **kwargs):
        if self.youtube is True:
            video, details, fps, error = self.is_youtube(url=path_or_url)
        else:
            video, details, fps,  error = self.isno_youtube(path=path_or_url)

        video_data = None 
        
        if error is None:
            kwargs["fps"] = fps
            video_data = self.yolovo_video_track_demo(video=video, df=self.df, 
                                        details=details, response=self.response, colors=self.colors, 
                                        model=self.model, tracker=tracker, save=save, **kwargs)
        else: print(error)

        return video_data
