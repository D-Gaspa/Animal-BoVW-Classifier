from termcolor import colored
from preprocessing.enhancement_scripts.run_pipeline import main as Enhancer
from classification.bovw import main as Train
from preprocessing.enhancement_scripts.size_transform import ImageEnhancer
from preprocessing.filter_transformations.apply_filters import ApplyFilters
from pathlib import Path
import os

def main():
    loop = True
    verifyFPath = f'{Path.cwd()}\\docs\\.gitkeep'
    resFPath = f'{Path.cwd()}\\docs\\.smallImg'

    print(colored("Welcome to the Animal images enhancer & classifier", "white", attrs=['bold']))

    while(loop):
        if os.path.exists(verifyFPath) == False:
            print(colored("\nPlease select one of the following options:", "white"))
            print(colored("\n1.Enhance Image dataset", "light_green"))
            print(colored("\n2.Classify image dataset", "red"))
            print(colored("\n3.Exit application", "light_blue"))
        else:
            print(colored("\nPlease select one of the following options:", "white"))
            print(colored("\n1.Enhance Image dataset", "light_green"))
            print(colored("\n2.Classify image dataset", "light_green"))
            print(colored("\n3.Exit application", "light_blue"))
        
        res = input()

        if res == '1':
            Enhancer()
            file = open(verifyFPath, mode= 'w')
            file.close()
            print(colored("\n\nThe image enhancing has been performed.", "light_green"))
        elif res == '2':
            if os.path.exists(verifyFPath) == False:
                print(colored("\nYou cannot classify images if they are not resized and enhanced before.", "red"))
            else:
                base_path = f'{Path.cwd()}\\data\\raw_dataset'
                filtered_path = f'{Path.cwd()}\\data\\filtered_images'
                if os.path.exists(resFPath) == False:
                    for class_name in os.listdir(base_path):
                        class_path = os.path.join(base_path, class_name)
                        resized_class_path = os.path.join(f'{Path.cwd()}\\data\\small_images', class_name)
                        smallResize = ImageEnhancer(class_path, resized_class_path)
                        smallResize.small_images(350)
                    file = open(resFPath, mode= 'w')
                    file.close()
                smallImgPath = f'{Path.cwd()}\\data\\small_images'
                filters_to_apply = ["histogram_equalization", "noise_reduction"]
                apply_filters = ApplyFilters(smallImgPath, filtered_path, filters_to_apply)
                apply_filters.apply()
                Train()
                print(colored("\n\nThe image enhancing has been performed.", "light_green"))
        elif res == '3':
            loop = False
            print(colored("\nExiting application...", "light_grey"))
            
if __name__ == "__main__":
    main()