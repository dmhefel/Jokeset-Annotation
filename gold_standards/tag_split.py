#need to install python-bs4 and lxml

from bs4 import BeautifulSoup
import re
import os

#make folders for every joke type if not already present
path = os.getcwd()

try:
    # folder names
    wordplay = path + "/wordplay"
    reference = path + "/reference"
    shock = path + "/shock"
    character = path + "/character"
    focus = path + "/focus"

    os.mkdir(wordplay)
    os.mkdir(reference)
    os.mkdir(shock)
    os.mkdir(character)
    os.mkdir(focus)
except FileExistsError:
    print("Joke category folders already exist.  No new folders created.")
except OSError:
    print("Creation of the directory %s failed.  \nCheck if the slashes in the 'folder names' section are in the right direction for your operating system,\nand delete the wordplay folder if it was created." % path)
    quit()

#need to loop this for all golden files
for file in os.listdir("."):
    if file.endswith(".xml"):
        # parse the golden file
        print(file)
        file = open(file, encoding='utf-8')
        contents = file.read()
        file.close()
        soup = BeautifulSoup(contents, 'xml')

        tags = soup.find('TAGS')
        jokes = soup.find_all('TEXT')

        # get the tag/item number pairs
        tags = str(tags).split("\n")
        tag_division = {}
        regex = re.compile(r"^<(\w*).*(ITEM +\d*)")

        for count in range(1,len(tags)-1):
            # print(tags[count])
            tag_and_id = regex.findall(tags[count])
            # print(tag_and_id)
            tag_division[tag_and_id[0][1]] = tag_and_id[0][0]

        # get the id/joke pairs
        regex = re.compile(r"(ITEM.*)(\n.*[\n]*)")
        joke_division = regex.findall(str(jokes))

        # make the files in the appropriate places
        for id,joke in joke_division:
            # get the appropriate folder path for the category
            file = path + "/" + tag_division[id]

            # make new file in relevant folder with id name and joke contents
            with open(os.path.join(file, id + ".txt"), 'w') as fp:
                fp.write(joke.strip())
