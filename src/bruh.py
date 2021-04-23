with open('data/captions.csv', 'r', encoding='utf8') as f1, open('data/Flickr30k/captions.txt', 'w') as f2:
    f2.write('image,caption')
    line = f1.readline()
    
    while True:
        line = f1.readline()
        if not line:
            break
        image, caption = line.strip().split('| ')[::2]
        caption = caption.replace('"', '')
        if ',' in caption:
            caption = '\"' + caption + '\"'
        
        f2.write("\n" + image + ',' + caption)



        """
        shjdkasdjghadnjgnasdxl
        """
        '''
        "sdsds""sdsdsd"sd"sd"sd"s'dS"D
        '''