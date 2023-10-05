from termcolor import colored
from preprocessing.enhancement_scripts.run_pipeline import main as Enhancer
from pathlib import Path
import os

def main():
    loop = True
    verifyFPath = f'{Path.cwd()}\\docs\\.gitkeep'

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
            file = open(verifyFPath, mode= 'w')
            file.close()
            Enhancer()
        elif res == '2':
            if os.path.exists(verifyFPath) == False:
                print(colored("\nYou cannot classify images if they are not resized and enhanced before.", "red"))
            else:
                print()
        elif res == '3':
            loop = False
            print(colored("\nExiting application...", "light_grey"))
            
if __name__ == "__main__":
    main()