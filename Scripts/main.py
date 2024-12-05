import dearpygui.dearpygui as dpg
from pathlib import Path
import numpy as np
import os
import time
import glob
from PIL import Image
import torch
import threading
from ultralytics import YOLO
import cv2
import pywt

#from AI import AI
# Window dimensions
winw = 1200
winh = 850
selected_Path = [] 
selected_files = [] #copy filename from path
focus_treshold = 1000
auto = True
started = False

class AppInit:        
    @staticmethod
    def CreateWindows():
        with dpg.window(label="Home",width=winw,
            height=winh,tag="whome",no_move=True,no_collapse=True,
            no_resize=True,no_title_bar=True
        ):
            # Menu bar
            with dpg.menu_bar():
                with dpg.menu(label="File", tag='file'):
                    dpg.add_menu_item(label="Open File", callback=lambda: AppInit.show_picker("input_file"))
                    dpg.add_menu_item(label="Open Folder", callback=lambda: AppInit.show_picker("input_folder"))
                with dpg.menu(label="Model", tag='model'):
                    with dpg.group(horizontal=True):
                        dpg.add_text("Use Custom")
                        dpg.add_checkbox(default_value=False,tag="custom_model",callback=Helper.checkbox)
                    dpg.add_menu_item(label="Pick Custom (!)",callback=lambda: AppInit.show_picker("model_select"),tag="pick_custom")
                dpg.add_button(label="Output Folder (!)", tag="out_folder_tag",callback=lambda: AppInit.show_picker("output_folder"))
                dpg.add_button(label="Options")
                dpg.add_button(label="Close", callback=lambda: Helper.closebox())
            
            # Main windows
            with dpg.group(horizontal=True):
                # Task box (Left Side)
                with dpg.child_window(width=winw-900, height=winh-80, border=True):
                    with dpg.collapsing_header(label="Tasks", default_open=True):
                        dpg.add_listbox(items=selected_files, tag="file_list", num_items=37,width=winw-916)
                    dpg.add_button(label="START",tag="op_start",width=winw-916,callback=lambda:Helper.start_main())
                    with dpg.group(horizontal=True):
                        #Total Tasks
                        win_width = 89
                        win_height = 38
                        with dpg.child_window(width=win_width, height=win_height, border=True):
                            dpg.add_input_text(enabled=False,width=win_width-16,tag="total_task",hint="Total")
                            
                        #Queue Tasks
                        with dpg.child_window(width=win_width, height=win_height, border=True):
                            dpg.add_input_text(enabled=False,width=win_width-16,tag="queue_task",hint="Queue")
                        #Finished Tasks
                        with dpg.child_window(width=win_width, height=win_height, border=True):
                            dpg.add_input_text(enabled=False,width=win_width-16,tag="finished_task",hint="Finish")
                    dpg.add_input_text(enabled=False,tag='sys_log',width=winw-917,hint="log")

                # Main window (Right Side)
                with dpg.child_window(width=winw-350, height=winh-504, border=True):
                    with dpg.group(horizontal=True):
                        with dpg.group(horizontal=False, tag='image_container'):
                            dpg.add_text('Image Preview',show=True)
                        with dpg.group(horizontal=False):
                            dpg.add_spacer(height=50)
                            dpg.add_text('No image selected', tag='preview_name')
                            dpg.add_spacer(height=20)
                            dpg.add_text('Path: nan', tag='preview_path')
                            with dpg.group(horizontal=True):
                                dpg.add_text('Dimentions: nan', tag='preview_dimention')
                                dpg.add_spacer(width=20)
                                dpg.add_text('Size: nan', tag='preview_size')
                            dpg.add_spacer(height=20)
                            dpg.add_text('Score: ', tag='preview_score')
                            dpg.add_text('Prediction: ', tag='preview_predict')
                            dpg.add_spacer(height=30)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Next",tag='next_task',callback=lambda: Helper.next_click())
                                dpg.add_button(label="Auto (0)",tag='auto_run',callback=lambda: Helper.autorun())
                                #dpg.add_button(label="Pause ||",tag='pause_task')
                                #dpg.add_button(label="Stop",tag='stop_task')

            # New window underneath the original window
            with dpg.child_window(width=winw-350, height=winh-500, border=True, pos=(winw-884, winh-482)):
                dpg.add_text("Detections")
                with dpg.group(horizontal=True, tag="image_group",):
                    dpg.add_text("test",show=False) #blank somthing so group works
                                  
                script_dir = os.path.dirname(os.path.abspath(__file__)) #default image
                Imagehandler.load_image_from_path(Path(script_dir)/"default.png")
                dpg.set_value("sys_log", "") #reset log
                
            # New window underneath the other window 
            with dpg.child_window(width=winw-350, height=winh-770, border=True, pos=(winw-884, winh-133)):
                with dpg.group(horizontal=True, tag="detection_list",):
                    dpg.add_text("test",show=False) #blank somthing so group works
                with dpg.group(horizontal=True, tag="detection_list2",):
                    dpg.add_text("test",show=False) #blank somthing so group works
                
        
        #Exit dialog window
        with dpg.window(label="Exit Dialog", width=300, height=150, 
                        show=False, tag="close_dialog",
                        no_collapse=True, no_title_bar=False,
                        no_move=True, no_close=True
                        ,no_resize=True):
            
            dpg.add_spacer(height=5)
            dpg.add_text("Confirm abort and exit?", tag="cf_abort_txt", indent=61)
            dpg.add_text("Exiting", tag="exit_txt", indent=120, show=False)
            dpg.add_spacer(height=25)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=63)
                dpg.add_button(label="CANCEL", callback=lambda: Helper.hide('cancel'), tag="cancel_button")
                dpg.add_spacer(width=20)
                dpg.add_button(label="CONFIRM", callback=lambda: Helper.confirm_close(), tag="confirm_button")
            dpg.add_progress_bar(label="Progress", tag="progress_bar", show=False, indent=45, width=200)
            
            
            
    @staticmethod
    def Picker():
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=AppInit.file_callback,
            tag="file_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".*", color=(234, 123, 45, 255))
            dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
            dpg.add_file_extension(".jpeg", color=(0, 255, 0, 255))
            dpg.add_file_extension(".png", color=(0, 255, 0, 255))
            # Uppercase
            dpg.add_file_extension(".JPG", color=(0, 255, 0, 255))
            dpg.add_file_extension(".JPEG", color=(0, 255, 0, 255))
            dpg.add_file_extension(".PNG", color=(0, 255, 0, 255))
            dpg.add_file_extension(".pt", color=(255, 0, 255, 255))

    @staticmethod
    def show_picker(mode):  # for picker selection
        is_folder = mode == "input_folder" or mode == "output_folder"
        dpg.configure_item("file_dialog", directory_selector=is_folder, user_data=mode)
        dpg.show_item("file_dialog")

    @staticmethod
    def file_callback(sender, app_data, user_data):
        # Select Files
        if user_data == "input_file":
            selections = app_data.get('selections', {})
            for selection in selections.values():
                if selection.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check if image
                    file_name = os.path.basename(selection)
                    selected_Path.append(selection)
                    selected_files.append(file_name)
            dpg.configure_item("file_list", items=selected_files)
        
        # Select Folder
        elif user_data == "input_folder":
            folder_path = app_data['file_path_name']
            dpg.set_value("sys_log", f"Folder selected: {folder_path}")
            image_files = glob.glob(os.path.join(folder_path, "*.*"), recursive=True)
            for image_file in image_files:
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_name = os.path.basename(image_file)
                    selected_Path.append(image_file)
                    selected_files.append(file_name)
            dpg.configure_item("file_list", items=selected_files)
        
        # Output Folder
        elif user_data == "output_folder":
            global output_path
            output_path = app_data['file_path_name']
            dpg.set_item_label("out_folder_tag","Output Folder")
            dpg.set_value("sys_log", f"Folder selected: {output_path}")
        
        # Model Select
        elif user_data == "model_select":
            selections = app_data.get('selections', {})
            if len(selections) != 1:
                dpg.set_value("sys_log", "Error : Select only 1 model")
                return
            
            if list(selections.keys())[0].lower().endswith(('.pt')):
                dpg.set_item_label("pick_custom","Pick Custom :)")
                global custom_model_path 
                custom_model_path = list(selections.values())[0]
                dpg.set_value("sys_log", f"Model selected: {os.path.basename(custom_model_path)}")
            else:
                dpg.set_value("sys_log", f"Invalid file type! (.pt only!)")
                
    @staticmethod
    def init_viewport_spec():
        dpg.create_viewport(title='Photosort V0.8', width=winw, height=winh, resizable=False)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("whome", True)

        # Center the child window
        dpg.set_item_pos("close_dialog", (winw // 2 - 150, winh // 2 - 100))
        
class Helper:
    
    @staticmethod
    def closebox():
        dpg.show_item("close_dialog")
        dpg.focus_item("close_dialog")
       
    @staticmethod
    def hide(mode):
        if mode == "cancel":
            dpg.set_primary_window("close_dialog", False)
            dpg.configure_item("close_dialog", show=False)

    @staticmethod
    def confirm_close():
        dpg.configure_item("exit_txt", show=True)
        dpg.configure_item("cf_abort_txt", show=False)
        dpg.configure_item("cancel_button", show=False)
        dpg.configure_item("confirm_button", show=False)
        dpg.configure_item("progress_bar", show=True)
        for i in range(101):
            dpg.set_value("progress_bar", i / 100.0)
            time.sleep(0.01)
        dpg.stop_dearpygui()
    
    @staticmethod
    def checkbox():
        if (dpg.get_value("custom_model")):
            try: 
                print(custom_model_path)
            except:
                dpg.set_value("sys_log", f"Select Model Path")
                dpg.set_item_label("pick_custom","Pick Custom (!)")
                return
           
    @staticmethod
    def autorun():
        global auto
        if (not auto):
            auto = True
            dpg.set_item_label("auto_run", "Auto (0)")
            dpg.set_value("sys_log", f"Set Auto to {auto}")
        else:
            auto = False
            dpg.set_item_label("auto_run", "Auto ( )")
            dpg.set_value("sys_log", f"Set Auto to {auto}")
    
    @staticmethod
    def next_click():
        dpg.set_value("sys_log", "Next button Clicked")
        Helper.start_main()
    
    @staticmethod
    def start_main():
        
        # Check if the queue is empty at the start
        if len(selected_files) == 0:
            dpg.set_value("sys_log", f"L: No photos in queue")
            return
        
        #check if output path is selected
        try: 
            print(output_path)
        except:
            dpg.set_value("sys_log", f"No Output Selected")
            return
        dpg.set_item_label("op_start", "RUNNING")
        
        Total = len(selected_Path) 
        for finished in range(Total):
            # Break the loop if the queue becomes empty
            if len(selected_files) == 0:
                break
            
            print(f"Processing file {finished + 1} of {Total}")
            
            # Update progress indicators
            dpg.set_value("total_task", f"T: {Total}")
            dpg.set_value("queue_task", f"Q: {len(selected_files)}")
            dpg.set_value("finished_task", f"F: {finished + 1}")
            
            #print(selected_Path[0])
            Imagehandler.load_image_from_path(selected_Path[0])
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            ai = AI()
            custom = dpg.get_value("custom_model")
            
            if (custom):
                try: 
                    print(custom_model_path)
                except:
                    dpg.set_value("sys_log", f"No Model Selected")
                    return
                run_pass = ai.run_yolo_inference(
                model_path=Path(custom_model_path),
                image_path=selected_Path[0],  # Replace with the actual image path
                conf_thresh=0.8,
                max_height=300,
                max_width=400)
            else:
                run_pass = ai.run_yolo_inference(
                    model_path=Path(script_dir)/"default.pt",
                    image_path=selected_Path[0],  # Replace with the actual image path
                    conf_thresh=0.8,
                    max_height=300,
                    max_width=400)
            
            if run_pass:
                selected_files.pop(0)
                selected_Path.pop(0)
            
            dpg.configure_item("file_list", items=selected_files)
            
            if (not auto):
                dpg.set_item_label("op_start", "START")
                return
            
        # Final check to update UI if all files are processed
        if len(selected_files) == 0:
            dpg.set_value("sys_log", f"L: All photos processed")
            dpg.set_value("queue_task", f"Q: 0")
            dpg.set_value("total_task", f"T: 0")
        dpg.set_item_label("op_start", "START")
        
    @staticmethod
    def assess_photo_focus(image):
        gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
        gray = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2GRAY)
        coeffs = pywt.wavedec2(gray, 'haar', level=2)
        cH, cV, cD = coeffs[1]
        energy = np.sum(np.square(cH)) + np.sum(np.square(cV)) + np.sum(np.square(cD))
        
        return energy
    
    @staticmethod
    def Focus_compute(focus_score):
        print(focus_score)
        
class Imagehandler:
    
    @staticmethod
    def load_image_from_path(file_path):
        
        if not os.path.isfile(file_path):
            dpg.set_value('preview_name', "Invalid file path.")
            return

        # Load the image and get its texture ID
        image_info = Imagehandler.add_and_load_image(file_path, 400, 300)  # Adjust max width and height as needed
        if image_info:
            
            # Remove previous image if exists
            if dpg.does_item_exist('current_image'):
                dpg.delete_item('current_image')

            # Add new image
            dpg.add_image(image_info.get('texture'), tag='current_image', parent='image_container')
            dpg.set_value('preview_name', f'File: {os.path.basename(file_path)}')
            dpg.set_value('preview_path',f'Path: {file_path}')
            dpg.set_value('preview_dimention',f'Dimention: {image_info.get("sizex")} * {image_info.get("sizey")}')
            dpg.set_value('preview_size',f'Size: {image_info.get("f_size")} MBs')
        else:
            dpg.set_value('preview_name', "Failed to load image.")


    @staticmethod
    def add_and_load_image(image_path, max_width, max_height):
        
        #clear last texture for ram management
        if hasattr(Imagehandler, 'last_texture'):
            try:
                dpg.delete_item(Imagehandler.last_texture)
            except Exception as e:
                print(f"Error deleting previous texture: {e}")
                
        dpg.set_value("sys_log", f"Loading image: {image_path}")
        #print(f"Loading image: {image_path}")
        try:
            image = Image.open(image_path)
            aspect_ratio = image.width / image.height
            
            #save original size for preview
            original_w = image.width 
            original_h = image.height
            original_s = round(os.path.getsize(image_path) / (1024 * 1024), 3)
            
            # Adjust width and height to fit within max dimensions
            new_height = image.height
            new_width = image.width
            if image.width > max_width or image.height > max_height:
                if image.width / max_width > image.height / max_height:
                    new_width = max_width
                    new_height = int(max_width / aspect_ratio)
                else:
                    new_height = max_height
                    new_width = int(max_height * aspect_ratio)
                
                # Resize the image
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert the image to RGBA format
            image = image.convert("RGBA")
            data = image.tobytes()

            # Normalize the image data
            data = [d / 255.0 for d in data]

            with dpg.texture_registry():
                texture_id = dpg.add_static_texture(new_width, new_height, data)

            # Store the current texture ID for potential deletion next time
            Imagehandler.last_texture = texture_id

            return {'texture': texture_id ,'sizex' : original_w ,'sizey' : original_h,'f_size' : original_s}
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
class AI:
    # Static class variables
    image_count = 0
        
    def run_yolo_inference(
        self,
        model_path: str,
        image_path: str,
        conf_thresh: float,
        max_width: int,
        max_height: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ):
        
        #clear previous texture_id and texture_tag
        if hasattr(AI, 'created_textures'):
            try:
                for textures in AI.created_textures:
                    dpg.delete_item(textures)
                AI.created_textures.clear()
            except Exception as e:
                print(f"Error deleting previous texture: {e}")
        else:
            AI.created_textures = []

        try:
            # Load YOLO model
            print(model_path)
            if not os.path.exists(model_path):
                dpg.set_value("sys_log", f"Model file not found: {model_path}")
                return False
            model = YOLO(model_path)

            # Load the input image
            img = cv2.imread(image_path)

            dpg.set_value("sys_log",f"Running YOLO on {image_path}")

            # Run inference
            results = model(image_path, conf=conf_thresh, device=device)
            
             #clear old list
            children = dpg.get_item_children("detection_list", 1)
            for child in children:
                dpg.delete_item(child)
            children = dpg.get_item_children("detection_list2", 1)
            for child in children:
                dpg.delete_item(child)
            
            if not results or all(len(r.boxes) == 0 for r in results):
                dpg.set_value("sys_log", f"No detections found for {os.path.basename(image_path)}")
                dpg.add_text(f"No detection found", parent="detection_list")
                time.sleep(0.05)
                return True
            else:
                dpg.set_value('sys_log', f"Processing {os.path.basename(image_path)} pass")
            total_scores={}
            
            count = 0
            # Process detections
            for result in results:
                boxes = result.boxes

                for box in boxes:
                    try:
                       # Get bounding box coordinates and class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        class_name = model.names[cls]
                        cropped_subject = img[y1:y2, x1:x2]
                        height, width = cropped_subject.shape[:2]
                        #print(f"Width: {width}, Height: {height}")
                        img_size = width * height
                        focus = round(Helper.assess_photo_focus(cropped_subject)/10000,2)
                        #print(f"Wavelet score: {focus}")
                        if (count < 4):
                            dpg.add_text(f"[{class_name}, Focus: {focus}], ", parent="detection_list")
                        else:
                            dpg.add_text(f"[{class_name}, Focus: {focus}], ", parent="detection_list2")
                        count+=1
                        # Create or update the total_scores dictionary
                        if class_name not in total_scores:
                            total_scores[class_name] = set()

                        # Add a tuple of (img_size, focus_score) to the set
                        total_scores[class_name].add((img_size, round(focus/10000, 2)))
                        
                        border_size = 10
                        if (focus >= focus_treshold):
                            border_color = (0, 255, 0)  # Green (BGR format)
                        elif focus >= focus_treshold/2:
                            border_color = (0, 165, 255)  # Orange
                        else:
                            border_color = (0, 0, 255)  # Orange
                        image_with_border = cv2.copyMakeBorder(
                            cropped_subject,
                            border_size, border_size, border_size, border_size,
                            cv2.BORDER_CONSTANT,
                            value=border_color
                        )
                        
                        # Resize cropped subject while maintaining aspect ratio
                        image = Image.fromarray(cv2.cvtColor(image_with_border, cv2.COLOR_BGR2RGB))
                        aspect_ratio = image.width / image.height
                        new_width, new_height = image.width, image.height
                        if new_width > max_width or new_height > max_height:
                            if new_width / max_width > new_height / max_height:
                                new_width = max_width
                                new_height = int(max_width / aspect_ratio)
                            else:
                                new_height = max_height
                                new_width = int(max_height * aspect_ratio)
                            image = image.resize((new_width, new_height), Image.LANCZOS)

                        # Convert the image to RGBA and prepare texture data
                        image = image.convert("RGBA")
                        data = list(image.tobytes())

                        # Normalize the image data
                        data = [d / 255.0 for d in data]
                        
                        texture_tag = f"texture_{os.path.splitext(os.path.basename(image_path))[0]}_{AI.image_count}"
                        AI.image_count += 1
                        
                        # Register and add texture to DearPyGui
                        with dpg.texture_registry():
                            texture_id = dpg.add_static_texture(new_width, new_height, data)
                            AI.created_textures.append(texture_id) # append texture_id for deletion

                        dpg.add_image(texture_id, tag=texture_tag, parent='image_group')
                        AI.created_textures.append(texture_tag) # append texture_tag for deletion
                    except Exception as box_err:
                        dpg.set_value("sys_log", f"Error processing bounding box: {box_err}")
                        return False
            return True
        except Exception as e:
            dpg.set_value("sys_log", f"Error during YOLO inference: {e}")
            return False


          
def main(): 
    # Initialize DearPyGui context
    dpg.create_context()

    # Initialize App
    AppInit.CreateWindows()
    AppInit.Picker()
    AppInit.init_viewport_spec()

    # Start GUI loop
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
