import cv2
import configparser
import dearpygui.dearpygui as dpg
import glob
import logging
from logging.handlers import RotatingFileHandler
import math
import numpy as np
import os
import shutil
import torch
import time
from ultralytics import YOLO
import pywt
from PIL import Image
from pathlib import Path


# Window dimensions
winw = 1200
winh = 850
selected_Path = [] 
selected_files = [] #copy filename from path
sharp_files = {}
blur_files = {}
unsure_files = {}
focus_treshold = 1500
auto = False
copy = False
config_file_path = f'{Path.cwd()}\Sortie.ini'
sharp = 5
blur = 5
unsure = 5
version = "v1.8.5 Release"

class AppInit:        
    @staticmethod
    def CreateWindows():
        average_line_size = 100  # Adjust this based on your actual log line size
        max_lines = 1000 
        max_bytes = average_line_size * max_lines
        handler = RotatingFileHandler(f'{Path.cwd()}\Sortie.log', maxBytes=max_bytes, backupCount=1)
        logging.basicConfig(handlers=[handler], level=logging.DEBUG,
                            format='%(asctime)s - %(message)s',
                            datefmt='%H:%M:%S')
        
        with dpg.window(label="Home",width=winw,
            height=winh,tag="whome",no_move=True,no_collapse=True,
            no_resize=True,no_title_bar=True
        ):
            # Menu bar
            with dpg.menu_bar():
                with dpg.menu(label="File", tag='file'):
                    dpg.add_menu_item(label="Open File", callback=lambda: AppInit.show_picker("input_file"))
                    dpg.add_menu_item(label="Open Folder", callback=lambda: AppInit.show_picker("input_folder"))
                    dpg.add_menu_item(label="Clear Queue", callback=lambda: (selected_files.clear(),selected_Path.clear(),dpg.configure_item("file_list", items=selected_files)))
                with dpg.menu(label="Model", tag='model'):
                    with dpg.group(horizontal=True):
                        dpg.add_text("Use Custom")
                        dpg.add_checkbox(default_value=False,tag="custom_model")
                    dpg.add_menu_item(label="Custom Model (!)",callback=lambda: AppInit.show_picker("model_select"),tag="pick_custom")
                with dpg.menu(label="Output", tag='out_folder'):
                    with dpg.group(horizontal=True):
                        dpg.add_text("Use Custom")
                        dpg.add_checkbox(default_value=False,tag="custom_output")
                    dpg.add_menu_item(label="Output Folder (!)", tag="out_folder_tag",callback=lambda: AppInit.show_picker("output_folder"))
                dpg.add_button(label="Options",callback=lambda:(dpg.show_item("option_window"),dpg.focus_item("option_window")))
                dpg.add_button(label="About",callback=lambda:dpg.configure_item("about_window",show=True))
                dpg.add_button(label="Close", callback=lambda: (dpg.show_item("close_dialog"),dpg.focus_item("close_dialog")))
            
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
                                dpg.add_button(label="Next",tag='next_task',callback=lambda: (Helper.writeToLog("Next button Clicked") , Helper.start_main()))
                                dpg.add_button(label="Auto (0)",tag='auto_run',callback=lambda: Helper.autorun())
                            dpg.add_spacer(height=5)
                            dpg.add_button(label="View Summary",tag="view_summary",show=False,callback=lambda: (dpg.configure_item("stat_window", show=True), Helper.SummaryBack(),dpg.delete_item('preview_img')if dpg.does_item_exist('preview_img') else None))
                            dpg.add_spacer(height=25)
                            dpg.add_progress_bar(show=False,label="Progress",tag="image_progress",width=250)

            # New window underneath the original window
            with dpg.child_window(width=winw-350, height=winh-500, border=True, pos=(winw-884, winh-482)):
                dpg.add_text("Detections")
                with dpg.group(horizontal=True, tag="image_group",):
                    dpg.add_text("test",show=False) #blank somthing so group works
                                  
                script_dir = Path.cwd() #default image
                Imagehandler.load_image_from_path(Path(script_dir)/"default.png")
                Helper.writeToLog("") #clear last log
                
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
        
        #Option Window
        with dpg.window(label="Options Window", width=450, height=220, 
                show=False, tag="option_window",
                no_collapse=True, no_title_bar=False,
                no_move=True, no_close=True
                ,no_resize=True ,pos=[(winw/2)-100,(winh/2)-200]):
            dpg.add_spacer(height=2)
            with dpg.group(horizontal=True):
                dpg.add_text("Copy Files instead of moving", indent=10)
                dpg.add_checkbox(tag="copy_file",indent=220)
            with dpg.group(horizontal=True):
                dpg.add_text("Display stats after finish", indent=10)
                dpg.add_checkbox(tag="display_stats",indent=220)
            dpg.add_spacer(height=10)
            dpg.add_text("Detection threshold (default 0.8) (lower is more sensitive)", indent=10)
            dpg.add_slider_double(tag='conf',default_value=0.5, min_value=0.0, max_value=1.0,indent=70)
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(label="View Logs History",callback=lambda:Helper.Openlog())
                dpg.add_button(label="CLear Log history",callback=lambda:(Helper.Clearlog()))
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save",tag="save_options",width=50,callback=lambda:Helper.SaveOption())
                dpg.add_button(label="Back",tag="option_back",width=50,callback=lambda:Helper.hide("option"))
            
            config = AppInit.ConfigCall()
            if (config['Settings'].getboolean('copy_f')):
                dpg.set_value("copy_file",True)
            else:
                dpg.set_value("copy_file",False)
            if (config['Settings'].getboolean('stats_d')):
                dpg.set_value("display_stats",True)
            else:
                dpg.set_value("display_stats",False)
            dpg.set_value("conf",float(config['Settings'].get('tresh')))
        
        with dpg.window(label="Logs History", width=800, height=500, 
                show=False, tag="log_window",
                no_collapse=True, no_title_bar=False,
                no_move=False, no_close=False
                ,no_resize=False):
            dpg.add_text("", tag="log_text")
            dpg.add_button(label="Back",callback=lambda:(dpg.configure_item("log_window", show=False),dpg.set_primary_window("log_window",False)))
        
        with dpg.window(label="Analytics", width=853, height=500, 
                show=False, tag="stat_window",
                no_collapse=True, no_title_bar=False,
                no_move=False, no_close=False
                ,no_resize=True,pos=[(winw/2)-400,(winh/2)-400]):
            global sharp, blur, unsure
            labels = ["Sharp", "Blur", "Unsure"]
            values = [sharp,blur,unsure]
            with dpg.group(horizontal=True,tag='summary_1'):
                with dpg.group(horizontal=False,tag="summary_panel"):
                    dpg.add_text("Summary")
                    with dpg.child_window(width=winw-1000,height=150,tag="stat_pan"):
                        dpg.add_text(f"Total Processed:",tag="total")
                        dpg.add_spacer(height=5)
                        dpg.add_text(f"Sharp Photos:",tag="sharp")
                        dpg.add_text(f"Blur Photos:",tag="blur")
                        dpg.add_text(f"Unsure Photos:",tag="unsure")
                        dpg.add_spacer(height=5)
                        dpg.add_text(f"Sharp Percentage: ",tag='percent')
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Next",callback=lambda:Helper.SummaryNext())
                        dpg.add_button(label="Close",callback=lambda:dpg.configure_item('stat_window',show=False))
                with dpg.group(horizontal=False):
                    dpg.add_text("Model Info")
                    with dpg.child_window(width=winw-1000,height=150,tag="Model_info"):
                        if (dpg.get_value("custom_model")):
                            dpg.add_text(f"Model Name: {os.path.basename(custom_output_path)}")
                        else:
                            dpg.add_text(f"Model Name: default.pt")
                        dpg.add_text("Yolo version v.11")
                        config = configparser.ConfigParser()
                        config.read(config_file_path)
                        dpg.add_spacer(height=5)
                        dpg.add_text(f"Minimum Confidence")
                        dpg.add_text(f"threshold: {float(config['Settings'].get('tresh'))}")
                with dpg.group(horizontal=False,tag="pie_panel"):
                    dpg.add_text("Pie-chart")
                    with dpg.plot(label="Detection Percentage", height=400, width=400):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis)
                        dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis")
                        dpg.add_pie_series(tag="pie_series",x=0.5,y=0.5,radius=0.4,values=values,labels=labels,parent="y_axis")
            with dpg.group(horizontal=True,tag='summary_2',show=False):  
                with dpg.group(horizontal=False):
                    dpg.add_text("Image Preview")
                    with dpg.child_window(width=winw-1000,height=160,tag="summary_image"):
                        dpg.add_text("Image Preview",show=False)   
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Close",callback=lambda:dpg.configure_item("stat_window", show=False))
                        dpg.add_button(label="Back",callback=lambda:Helper.SummaryBack())
                with dpg.group(horizontal=False,show=True,tag="sharp_panel"):
                    dpg.add_text("Sharp photos")
                    dpg.add_listbox(default_value="",items=list(sharp_files.keys()), callback=Helper.on_sharp_selected, tag="sharp_list", num_items=25,width=winw-1000)
                with dpg.group(horizontal=False,show=True,tag="blur_panel"):
                    dpg.add_text("Blur photos")
                    dpg.add_listbox(default_value="",items=list(blur_files.keys()),callback=Helper.on_blur_selected, tag="blur_list", num_items=25,width=winw-1000)
                with dpg.group(horizontal=False,show=True,tag="unsure_panel"):
                    dpg.add_text("Unsure photos")
                    dpg.add_listbox(default_value="",items=list(unsure_files.keys()),callback=Helper.on_unsure_selected, tag="unsure_list",num_items=25,width=winw-1000)
        
        with dpg.window(label="About", width=450, height=180, 
                show=False, tag="about_window",
                no_collapse=True, no_title_bar=False
                ,no_resize=True ,pos=[(winw/2)-100,(winh/2)-200]):
            dpg.add_text(f"S0rtieAI version {version}")
            dpg.add_spacer(height=5)
            dpg.add_text("Developed by")
            dpg.add_text("Lead programmer : Tanabodhi Mukura",indent=10)
            dpg.add_spacer(height=5)
            dpg.add_text("Developement Timeline")
            dpg.add_text("26 Nov - 11 Dec 2024",indent=10)
            
            
               
                
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
            Helper.writeToLog(f"Folder selected: {folder_path}")
            image_files = glob.glob(os.path.join(folder_path, "*.*"), recursive=True)
            for image_file in image_files:
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_name = os.path.basename(image_file)
                    selected_Path.append(image_file)
                    selected_files.append(file_name)
            dpg.configure_item("file_list", items=selected_files)
        
        # Output Folder
        elif user_data == "output_folder":
            output_path = app_data.get('selections', {})
            Helper.writeToLog(output_path)
            if len(output_path) > 1:
                Helper.writeToLog("Error : Select only 1 model")
                return

            dpg.set_item_label("out_folder_tag","Output Folder")
            global custom_output_path
            custom_output_path = list(output_path.values())[0]
            Helper.writeToLog(f"Folder selected:  {os.path.basename(custom_output_path)}")
            Helper.writeToLog(f"Custom output = {custom_output_path}")
           
        
        # Model Select
        elif user_data == "model_select":
            selections = app_data.get('selections', {})
            if len(selections) != 1:
                Helper.writeToLog("Error : Select only 1 model")
                return
            
            if list(selections.keys())[0].lower().endswith(('.pt')):
                dpg.set_item_label("pick_custom","Custom Model")
                global custom_model_path 
                custom_model_path = list(selections.values())[0]
                Helper.writeToLog(f"Model selected: {os.path.basename(custom_model_path)}")
            else:
                Helper.writeToLog(f"Invalid file type! (.pt only!)")
                
    @staticmethod
    def init_viewport_spec():
        dpg.create_viewport(title=f'S0rtieAI {version}', width=winw, height=winh, resizable=False,y_pos=0)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("whome", True)
        dpg.set_item_pos("whome", [0, 0])
        # Center the child window
        dpg.set_item_pos("close_dialog", (winw // 2 - 150, winh // 2 - 100))
    
    @staticmethod
    def ConfigCall():
        config = configparser.ConfigParser()
        if os.path.exists(config_file_path):
            config.read(config_file_path)
            Helper.writeToLog("Configuration loaded from file.")
            global auto
            if (config["Settings"].getboolean('Auto')):
                dpg.set_item_label("auto_run", "Auto (0)")
                auto = True
            else:
                dpg.set_item_label("auto_run", "Auto ( )")
                auto = False
        else:
            config['Settings'] = {
                'Auto': True,
                'copy_f': False,
                'stats_d': True,
                'tresh' : '0.8'
            }
            # Save the default configuration to the file
            with open(config_file_path, 'w') as configfile:
                config.write(configfile)
            Helper.writeToLog("Created missing config file")
        return config
    
class Helper:
    @staticmethod
    def on_sharp_selected(sender,app_data):
        file_name = app_data
        full_path = sharp_files[file_name]
        Imagehandler.preview_load_image_from_path(full_path)
    
    @staticmethod
    def on_blur_selected(sender,app_data):
        file_name = app_data
        full_path = blur_files[file_name]
        Imagehandler.preview_load_image_from_path(full_path)
    
    @staticmethod
    def on_unsure_selected(sender,app_data):
        file_name = app_data
        full_path = unsure_files[file_name]
        Imagehandler.preview_load_image_from_path(full_path)
        
    @staticmethod
    def SaveOption():
        cpf = dpg.get_value("copy_file")
        stats = dpg.get_value("display_stats")
        confi = dpg.get_value("conf")
        Helper.update_config("copy_f",cpf)
        Helper.update_config("stats_d",stats)
        Helper.update_config("tresh",confi)
        dpg.configure_item("option_window",show=False)
        
    @staticmethod
    def hide(mode):
        if mode == "cancel":
            dpg.set_primary_window("close_dialog", False)
            dpg.configure_item("close_dialog", show=False)

        if mode == "option":
            dpg.set_primary_window("option_window", False)
            dpg.configure_item("option_window", show=False)
            config = AppInit.ConfigCall()
            if (config['Settings'].getboolean('copy_f')):
                dpg.set_value("copy_file",True)
            else:
                dpg.set_value("copy_file",False)
            if (config['Settings'].getboolean('stats_d')):
                dpg.set_value("display_stats",True)
            else:
                dpg.set_value("display_stats",False)
            dpg.set_value("conf",float(config['Settings'].get('tresh')))
            
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
    def update_config(option, new_value):
        config = configparser.ConfigParser()
        config.read(config_file_path)
        new_value = str(new_value)
        # Update the specific value
        if 'Settings' in config and option in config['Settings']:
            config['Settings'][option] = new_value
            Helper.writeToLog(f"Updated {option} to {new_value}.")
        else:
            Helper.writeToLog(f"Option '{option}' not found in configuration.")
            return
        
        # Write the updated configuration back to the file
        with open(config_file_path, 'w') as configfile:
            config.write(configfile)

    @staticmethod
    def SummaryNext():
        dpg.configure_item("sharp_list", items=list(sharp_files.keys()))
        dpg.configure_item("blur_list", items=list(blur_files.keys()))
        dpg.configure_item("unsure_list", items=list(unsure_files.keys()))
        dpg.configure_item("summary_1", show=False)
        dpg.configure_item("summary_2", show=True)

    @staticmethod
    def SummaryBack():
        dpg.configure_item("summary_1", show=True)
        dpg.configure_item("summary_2", show=False)
        
    @staticmethod
    def autorun():
        global auto
        if (not auto):
            auto = True
            dpg.set_item_label("auto_run", "Auto (0)")
            Helper.writeToLog(f"Set Auto to {auto}")
            Helper.update_config('Auto', True)
        else:
            auto = False
            dpg.set_item_label("auto_run", "Auto ( )")
            Helper.writeToLog(f"Set Auto to {auto}")
            Helper.update_config('Auto', False)
            
    @staticmethod
    def writeToLog(msg):  #logging system
        dpg.set_value("sys_log",f"L: {msg}")
        logging.info(msg)
        print(f"L: {msg}")
      
    @staticmethod
    def Openlog():
        print("open")
        dpg.configure_item("log_window", show=True)
        #dpg.set_primary_window("log_window",True)
        with open(f'{Path.cwd()}/Sortie.log', 'r') as file:
            dpg.set_value("log_text", "".join(file.readlines()))

    @staticmethod
    def Clearlog():
        with open(f'{Path.cwd()}/Sortie.log', 'w'): # Opening the file in write mode and closing it immediately clears the file
            pass
        Helper.writeToLog("Log Cleared")
        
    @staticmethod
    def start_main():
        dpg.configure_item("stat_window", show=False)
        # Check if the queue is empty at the start
        if len(selected_files) == 0:
            Helper.writeToLog("No photos in queue")
            return

        #check if output path is selected and custom path is checked
        custom_out = dpg.get_value("custom_output")
        if (custom_out):
            try: 
                Helper.writeToLog(f"Using Custom output at {custom_output_path}")
            except:
                Helper.writeToLog("No Output Selected")
                return
        global sharp_files, blur_files, unsure_files
        dpg.configure_item("view_summary",show=False)
        sharp_files.clear()
        blur_files.clear()
        unsure_files.clear()
        old_dir = ""
        confidence = dpg.get_value("conf")
        Helper.writeToLog(f"Starting with Confidence threshold : {confidence}")
        
        dpg.set_item_label("op_start", "RUNNING")
        dpg.configure_item("image_progress", show = True)
        Total = len(selected_Path) 
        
        for finished in range(Total):
            dpg.set_value("preview_predict","Predict :") #reset prediction for every photo
            dpg.set_value("preview_score","Score: ") #reset prediction for every photo
            if (custom_out):
                output_path = os.path.dirname(custom_output_path)
            else:
                output_path = os.path.dirname(selected_Path[0])
                
            if old_dir != output_path:
                os.makedirs(Path(f"{output_path}/Sharp"), exist_ok=True)
                os.makedirs(Path(f"{output_path}/Blur"), exist_ok=True)
                os.makedirs(Path(f"{output_path}/Uncertain"), exist_ok=True)
                old_dir = output_path
            
            # Break the loop if the queue becomes empty
            if len(selected_files) == 0:
                break
            dpg.set_value("image_progress", 10/100.00)
            Helper.writeToLog(f"Processing file {finished + 1} of {Total}")
            
            # Update progress indicators
            dpg.set_value("total_task", f"T: {Total}")
            dpg.set_value("queue_task", f"Q: {len(selected_files)}")
            dpg.set_value("finished_task", f"F: {finished + 1}")
            
            Helper.writeToLog(f"Path = {selected_Path[0]}")
            Imagehandler.load_image_from_path(selected_Path[0])
            
            script_dir = Path.cwd()
            ai = AI()
            custom = dpg.get_value("custom_model")
            conf = round(dpg.get_value("conf"),2)
            if (custom):
                try: 
                    Path(custom_model_path)
                except:
                    Helper.writeToLog(f"No Model Selected")
                    dpg.set_item_label("op_start", "START")
                    dpg.set_value("image_progress", 0/100.00)
                    return
                run_result = ai.run_yolo_inference(
                model_path=Path(custom_model_path),
                image_path=selected_Path[0],  # Replace with the actual image path
                conf_thresh=conf,
                max_height=300,
                max_width=400)
            else:
                if not os.path.exists(Path(script_dir)/"default.pt"):
                    Helper.writeToLog(f"Default model not found")
                    dpg.set_item_label("op_start", "START")
                    dpg.set_value("image_progress", 0/100.00)
                    return
                run_result = ai.run_yolo_inference(
                    model_path=Path(script_dir)/"default.pt",
                    image_path=selected_Path[0],  # Replace with the actual image path
                    conf_thresh=conf,
                    max_height=300,
                    max_width=400)
            
            #AI return True and {data}   
            if run_result[1]:
                total_scores = run_result[1]

                dpg.set_value("preview_score",f"Score: {round(Helper.calculate_focus_score(total_scores),2)}")
                if total_scores:
                    if Helper.Focus_compute(total_scores):
                        dpg.set_value("preview_predict","Predict : Sharp")
                        new_path = f"{output_path}\Sharp"
                        sharp_files[selected_files[0]] = f"{new_path}\{selected_files[0]}"
                    else:
                        dpg.set_value("preview_predict","Predict : Blurry")
                        new_path = f"{output_path}\Blur"
                        blur_files[selected_files[0]] = f"{new_path}\{selected_files[0]}"
                    time.sleep(0.1)
                    
                    dpg.set_value("image_progress", 100/100.00)
                    time.sleep(0.025)
            else:
                #AI return True and {empty}
                new_path = f"{output_path}\\Uncertain"
                unsure_files[selected_files[0]] = f"{new_path}\{selected_files[0]}"
                
            config = AppInit.ConfigCall()
            if (config['Settings'].getboolean('copy_f')):
                shutil.copy(selected_Path[0], new_path)
                Helper.writeToLog(f"copy from {selected_Path[0]} to {new_path}")
            else:
                if not os.path.exists(new_path):
                    shutil.move(selected_Path[0], new_path,)
                else:
                    os.remove(selected_Path[0])
                Helper.writeToLog(f"move from {selected_Path[0]} to {new_path}")
            if run_result[0]:
                selected_files.pop(0) #POP TOP DATA (Already passed)
                selected_Path.pop(0)
            dpg.configure_item("file_list", items=selected_files)
                
            if (not auto):
                dpg.set_item_label("op_start", "START")
                dpg.set_value("image_progress", 0/100.00)
                dpg.configure_item("image_progress", show=False)
                return
            
        # Final check to update UI if all files are processed
        if len(selected_files) == 0:
            Helper.writeToLog(f"All photos processed")
            dpg.set_value("queue_task", f"Q: 0")
            dpg.set_value("total_task", f"T: 0")
            dpg.set_item_label("op_start", "START")
            dpg.set_value("image_progress", 0/100.00)
            dpg.configure_item("image_progress", show=False)
            if (config['Settings'].getboolean('stats_d')):
                Helper.writeToLog("Show stats")
                global sharp,blur,unsure,total
                dpg.set_value("sharp",f"Sharp Photos: {len(sharp_files.keys())}")
                dpg.set_value("blur",f"Blur Photos: {len(blur_files.keys())}")
                dpg.set_value("unsure",f"Unsure Photos: {len(unsure_files.keys())}")
                dpg.configure_item("pie_series",values=[len(sharp_files), len(blur_files), len(unsure_files)])
                total = len(sharp_files)+len(blur_files)+len(unsure_files)
                blur = len(blur_files.keys())
                unsure = len(unsure_files.keys())
                sharp = len(sharp_files.keys())
                dpg.set_value('percent',f"Sharp Percentage: {round((sharp/total)*100)}%")
                Helper.SummaryBack()
                dpg.configure_item("view_summary",show=True)
                dpg.configure_item("stat_window", show=True)
        
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
        size_treshold = 2000000
        print(focus_score)
        try:
            people = focus_score.get("person")
            #Phase 1 check for sharp human
            for person in people:
                if person[1] > focus_treshold:
                    Helper.writeToLog(f"Human focus: {person[1]}  PASS")
                    return True
            #Phase 2 check for small and a little blurry
            for person in people:
                if person[0]<size_treshold and person[1] > focus_treshold/2:
                    Helper.writeToLog(f"Human far focus: {person[1]} Size: {person[0]} PASS")
                    return True
            Helper.writeToLog("Human Subjects FAIL")
        except:
            Helper.writeToLog("No person detected")

        
        try:
            animals = focus_score.get("animal")             
            #Phase 1 check for sharp Animal
            for animal in animals:
                if animal[1] > focus_treshold:
                    Helper.writeToLog(f"Animal focus: {animal[1]}  PASS")
                    return True
            #Phase 2 check for small and a little blurry
            for person in people:
                if animal[0]<size_treshold and animal[1] > focus_treshold/2:
                    Helper.writeToLog(f"Animal far focus: {animal[1]} Size: {animal[0]} PASS")
                    return True
            Helper.writeToLog("Animal Subjects FAIL")
        except:
            Helper.writeToLog("No animal detected")

        try:
            Balls = focus_score.get("sports ball")
            for ball in Balls:
                if ball[1] > focus_treshold:
                    Helper.writeToLog(f"Ball focus: {animal[1]}  PASS")
                    return True
        except:
           Helper.writeToLog("No Balls detected")
        Helper.writeToLog("Focus Assesment : Blur")
        return False

    @staticmethod
    def calculate_focus_score(subjects):
        max_image_score = 100
        human_weight = 1.5  # Weight for human subjects
        focused_threshold = 1500  # Threshold for focused
        perfect_threshold = 2500  # Threshold for perfect focus

        total_score = 0
        total_possible_score = 0

        for subject_type, details in subjects.items():
            is_human = subject_type == 'person'
            
            for size, focus_score in details:
                # Normalize focus score to a scale of 0 to 1 based on the thresholds
                if focus_score <= focused_threshold:
                    normalized_focus_score = focus_score / focused_threshold
                else:
                    normalized_focus_score = 1 + (focus_score - focused_threshold) / (perfect_threshold - focused_threshold)
                    normalized_focus_score = min(normalized_focus_score, 2.0)  # Cap at 2.0 for perfect focus
                
                if is_human:
                    normalized_focus_score *= human_weight
                
                total_score += normalized_focus_score
                total_possible_score += 2.0 * (human_weight if is_human else 1)  # Adjusted to reflect the cap

        # Normalize the total score to a maximum of 100
        normalized_score = (total_score / total_possible_score) * max_image_score
        normalized_score = min(normalized_score, max_image_score)  # Ensure the score does not exceed 100
        return normalized_score


class Imagehandler:

    @staticmethod
    def load_image_from_path(file_path):
        
        if not os.path.isfile(file_path):
            dpg.set_value('preview_name', "Invalid file path.")
            return
        dpg.set_value("image_progress", 20/100.00)
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
            dpg.set_value("image_progress", 60/100.00)
        else:
            dpg.set_value('preview_name', "Failed to load image.")
    
    @staticmethod
    def preview_load_image_from_path(file_path):
        
        if not os.path.isfile(file_path):
            Helper.writeToLog("Invalid file path.")
            return
        # Load the image and get its texture ID
        image_info = Imagehandler.preview_load_img(file_path, int(400/2), int(300/2))  # Adjust max width and height as needed
        if image_info:
            # Remove previous image if exists
            if dpg.does_item_exist('preview_img'):
                dpg.delete_item('preview_img')

            # Add new image
            dpg.add_image(image_info.get('texture'), tag='preview_img', parent='summary_image')
        else:
            dpg.set_value('preview_name', "Failed to load image.")

    @staticmethod
    def preview_load_img(image_path, max_width, max_height):
        
        #clear last texture for ram management
        if hasattr(Imagehandler, 'last_preview'):
            try:
                dpg.delete_item(Imagehandler.last_preview)
            except Exception as e:
                Helper.writeToLog(f"Error deleting previous texture: {e}")
        Helper.writeToLog(f"Loading image: {os.path.basename(image_path)}")
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
            Helper.writeToLog(f"Error loading image: {e}")
            return None
    
    
    @staticmethod
    def add_and_load_image(image_path, max_width, max_height):
        
        #clear last texture for ram management
        if hasattr(Imagehandler, 'last_texture'):
            try:
                dpg.delete_item(Imagehandler.last_texture)
            except Exception as e:
                Helper.writeToLog(f"Error deleting previous texture: {e}")
        Helper.writeToLog(f"Loading image: {os.path.basename(image_path)}")
        dpg.set_value("image_progress", 40/100.00)
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
            dpg.set_value("image_progress", 50/100.00)
            # Normalize the image data
            data = [d / 255.0 for d in data]     
            with dpg.texture_registry():
                texture_id = dpg.add_static_texture(new_width, new_height, data)

            # Store the current texture ID for potential deletion next time
            Imagehandler.last_texture = texture_id       
            return {'texture': texture_id ,'sizex' : original_w ,'sizey' : original_h,'f_size' : original_s}
        except Exception as e:
            Helper.writeToLog(f"Error loading image: {e}")
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
                Helper.writeToLog(f"Error deleting previous texture: {e}")
        else:
            AI.created_textures = []

        try:
            # Load YOLO model
            Helper.writeToLog(model_path)
            if not os.path.exists(model_path):
                Helper.writeToLog(f"Model file not found: {model_path}")
                return (False,{})
            model = YOLO(model_path)
            dpg.set_value("image_progress", 70/100.00)
            # Load the input image
            img = cv2.imread(image_path)

            Helper.writeToLog(f"Running YOLO on {os.path.basename(image_path)}")
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
                Helper.writeToLog(f"No detections found for {os.path.basename(image_path)}")
                dpg.add_text(f"No detection found", parent="detection_list")
                dpg.set_value("image_progress", 80/100.00)
                time.sleep(0.05)
                return (True, {})
            else:
                Helper.writeToLog(f"Processing {os.path.basename(image_path)} pass")
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
                        img_size = width * height
                        focus = round(Helper.assess_photo_focus(cropped_subject)/10000,2)
                        if (count < 4):
                            dpg.add_text(f"[{class_name}, F: {focus}, S: {img_size}], ", parent="detection_list")
                        else:
                            dpg.add_text(f"[{class_name}, Focus: {focus}], ", parent="detection_list2")
                        count+=1
                        # Create or update the total_scores dictionary
                        if class_name not in total_scores:
                            total_scores[class_name] = set()

                        # Add a tuple of (img_size, focus_score) to the set
                        total_scores[class_name].add((img_size, focus))
                        
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
                        Helper.writeToLog(f"Error processing bounding box: {box_err}")
                        return (False,{})
            dpg.set_value("image_progress", 80/100.00)
            return (True,total_scores)
        except Exception as e:
            Helper.writeToLog(f"Error during YOLO inference: {e}")
            return (False,{})


          
def main(): 
    # Initialize DearPyGui context
    dpg.create_context()
    
    # Initialize App
    AppInit.CreateWindows()
    AppInit.Picker()
    AppInit.init_viewport_spec()
    AppInit.ConfigCall()
    # Start GUI loop
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
