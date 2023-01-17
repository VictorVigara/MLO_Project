import requests
response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')
with open(r'img.png','wb') as f:
   f.write(response.content)


pload = {'username':'Olivia','password':'123'}
response = requests.post('https://httpbin.org/post', data = pload)
