from termcolor import colored
from preprocessing.enhancement_scripts.run_pipeline import enhancer as Enhancer

def main():
    loop = True
    enhance = False

    print(colored("Welcome to the Animal images enhancer & classifier", "white", attrs=['bold']))

    while(loop):
        if enhance == False:
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
            enhance = True
            Enhancer()
        elif res == '2':
            if enhance == False:
                print(colored("\nYou cannot classify images if they are not resized and enhanced before.", "red"))
            else:
                print()
        elif res == '3':
            loop = False
            print(colored("\nExiting application...", "light_grey"))
            
if __name__ == "__main__":
    main()