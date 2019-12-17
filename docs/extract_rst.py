import re
import io
import os


def extract_package_structure(root_path="../pydeep/",ext=".py", exclude_unittest=True):
    """ Get all files and depth level

    :param root_path: Root directory
    :type root_path: string

    :return: List of modul
    :rtype:
    """
    structure = []
    for root, dirs, files in os.walk(root_path):
        for name in files:
            if name.endswith((ext, ext[1:])):
                fullpath= os.path.join(root, name)
                if not exclude_unittest or fullpath.find('test')<0:
                    structure.append([fullpath,
                                      fullpath.replace("/",".").replace("...","").replace(ext,""),
                                      name.replace(".py",""),
                                      fullpath.count("/")])
    return sorted(structure)

def print_classes_and_members(root_path,module_path,module_structure,module_name,depth):
    header_level = ["====================================================",
                    "----------------------------------------------------",
                    "````````````````````````````````````````````````````",
                    "''''''''''''''''''''''''''''''''''''''''''''''''''''",
                    "....................................................",
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
                    "****************************************************",
                    "++++++++++++++++++++++++++++++++++++++++++++++++++++",
                    "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"]
    if module_name == "__init__":
        pattern = re.compile('([^\.]+).__init__')
        module_name = pattern.findall(module_structure)[0]
        module_structure = module_structure.replace('.__init__','')
        depth -= 1
        print(module_name)
        print(header_level[depth])
        print("")
        print(".. automodule:: " + module_structure)
        print
    else:
        pattern = re.compile('( def|class|def)\s(\S+)\(')
        f = open(os.path.join(root_path, module_path), 'r')
        text = f.read()
        print(module_name)
        print(header_level[depth])
        print("")
        print(".. automodule:: " + module_structure)
        print
        current_class = ""
        for p in pattern.findall(text):
            if p[0] == "class":
                current_class = p[1]
                print p[1]
                print header_level[depth+1]
                print("")
                print(".. autoclass:: "+module_structure+"."+p[1])
                print("   :members:")
                print("   :private-members:")
                print("   :special-members: __init__")
                print("")
            elif p[0] == "def":
                print p[1]
                print header_level[depth+1]
                print("")
                print(".. automethod:: "+module_structure+"."+p[1])
                print("")
            #else:
            #    print p[1]
            #    print header_level[depth+2]
            #    print("")
            #    print("   .. automethod:: "+module_structure+"."+current_class+"."+p[1])
            #   print("")


struct = extract_package_structure()

for s in struct:
    #print s

    print_classes_and_members("",s[0],s[1],s[2],s[3])
