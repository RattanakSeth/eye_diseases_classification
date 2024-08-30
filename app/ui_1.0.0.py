
import tkinter
import tkinter.messagebox
import customtkinter
from util.eye_diseases_dataset import EyeDiseaseDataset
import os
from PIL import Image, ImageTk
from model.dcnn import DCNN_Model
import tensorflow as tf
import tkinter as tk
from tkinter import Label, filedialog
# from tkinter import filedialog
# from PIL import Image, ImageTk
# from tkinter import filedialog
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Eye Diseases Classification")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Eye Disease Classification", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        # self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=lambda: self.printGrapTraversal(Traversal.POSTORDER), text="Postorder")
        # self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        # self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=lambda: self.printGrapTraversal(Traversal.INORDER), text="Inorder")
        # self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        # self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, command=lambda: self.printGrapTraversal(Traversal.PREORDER), text="Preorder")
        # self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Select Your Model:", anchor="w")
        self.appearance_mode_label.grid(row=1, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["EffecientNetB3", "InceptionRestNetV3", "ResNet-18", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=2, column=0, padx=20, pady=(10, 10))



        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # create main entry and button
        # self.entry = customtkinter.CTkEntry(self, placeholder_text="CTkEntry")
        # self.entry.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        # self.main_button_1 = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        # self.main_button_1.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # create textbox
        # self.textbox = customtkinter.CTkTextbox(self, width=250, height=500, font=('Arial', 14))
        # self.textbox.grid(row=0, column=1, padx=(20, 20), pady=(20, 0), sticky="nsew")
        # self.textbox.grid_rowconfigure(4, weight=1)

        # Create right sidebar
        self.right_sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.right_sidebar_frame.grid(row=0, column=3, rowspan=4, sticky="nsew")
        self.right_sidebar_frame.grid_rowconfigure(4, weight=1)
        # self.tabview = customtkinter.CTkTabview(self, width=250)
        # self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        # self.tabview.grid_rowconfigure(4, weight=1)
        # self.tabview.add("CTkTabview")
        # self.tabview.add("Tab 2")
        # self.tabview.add("Tab 3")
        # self.tabview.tab("CTkTabview").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        # self.tabview.tab("Tab 2").grid_columnconfigure(0, weight=1)

        # self.optionmenu_1 = customtkinter.CTkOptionMenu(self.tabview.tab("CTkTabview"), dynamic_resizing=False,
        #                                                 values=["Value 1", "Value 2", "Value Long Long Long"])
        # self.optionmenu_1.grid(row=0, column=0, padx=20, pady=(20, 10))
        # self.combobox_1 = customtkinter.CTkComboBox(self.tabview.tab("CTkTabview"),
        #                                             values=["Value 1", "Value 2", "Value Long....."])
        # self.combobox_1.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.string_input_button = customtkinter.CTkButton(self.right_sidebar_frame, text="Open Insert", command=self.imageUploader)
        self.string_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))
        # self.label_tab_2 = customtkinter.CTkLabel(self.tabview.tab("Tab 2"), text="CTkLabel on Tab 2")
        # self.label_tab_2.grid(row=0, column=0, padx=20, pady=20)


        # Initial Model
        # self.cnnModel = DCNN_Model()
        # self.model = tf.keras.models.load_model("../training_2/efficientnetb3-Eye Disease-92.65.h5")#self.cnnModel.loadNewModel()

        # **Model Structure**
        # Start reading dataset
        data_dir = "dataset/eye_diseases_original_dataset"
        #
        try:
            dataEyeD = EyeDiseaseDataset(data_dir)
            # # Get splitted data
            train_df, valid_df, test_df = dataEyeD.split_data()
            print("test dataframe shape: ", test_df.shape)

            # Get Generators
            batch_size = 10
            train_gen, valid_gen, test_gen = dataEyeD.create_gens(train_df, valid_df, test_df, batch_size)

        except:
            print('Invalid Input')

        print("test_gen: ", test_gen)
        # self.eye_diseases_dataset = EyeDiseaseDataset(dataDir)
        # dataSplit = EyeDiseaseDataset(dataDir)
        # self.train_data, self.valid_data, self.test_data = dataSplit.split_()
        # print(self.train_data)

        # self.train_augmented, self.valid_augmented, self.test_augmented = dataSplit.augment_data(self.train_data, self.valid_data, self.test_data)

        self.my_image = customtkinter.CTkImage(light_image=Image.open(
            'dataset/eye_diseases_original_dataset/cataract/0_left.jpg'),
                                               dark_image=Image.open(
                                                   'dataset/eye_diseases_original_dataset/cataract/0_left.jpg'),
                                               size=(256, 256))  # WidthxHeight

        my_label = customtkinter.CTkLabel(self, text="", image=self.my_image)
        my_label.grid(row=0, column=1, padx=(20, 20), pady=(20, 0), sticky="nsew")
        my_label.grid_rowconfigure(4, weight=1)



    def open_input_dialog_event(self):
        self.imageUploader()
        # self.textbox.configure(state=customtkinter.NORMAL)
        #
        # dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="BST Insertion")
        # inputValue = dialog.get_input()
        # print("Binary Search tree Insert:", inputValue)
        # # print("type of this: ", int())
        # self.bst.insert(int(inputValue))
        # self.textbox.insert("insert", "Binary Search Tree Insert: " + inputValue + "\n")
        # self.textbox.configure(state=customtkinter.DISABLED)  # configure textbox to be read-only
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")

    def load_CNN_Model(self):
        self.cnnModel.loadNewModel()

    # image uploader function
    def imageUploader(self):
        f_types = [('Jpg Files', '*.jpg'), ('Png Files', '*.png')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        img = Image.open(filename)

        print(img)
        # if file is selected
        # if len(path):
        #
        #     img = Image.open(path)
        #     print("image: ", img)
            # img = img.resize((200, 200))
            # pic = ImageTk.PhotoImage(img)
            # return pic

            # print("path: ", path)

            # re-sizing the app window in order to fit picture
            # and buttom
            # app.geometry("560x300")
            # label.config(image=pic)
            # label.image = pic

        # if no file is selected, then we are displaying below message
        # else:
        #     print("No file is chosen !! Please choose a file.")


if __name__ == "__main__":
    app = App()
    app.mainloop()