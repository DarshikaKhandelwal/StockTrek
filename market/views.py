from django.shortcuts import render, HttpResponse
from django.core.files import File
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def index(request):
    return render(request, 'index.html')

url1='https://www.business-standard.com/markets-news'
response1 = requests.get(url1)
soup1 = BeautifulSoup(response1.text, 'html.parser')
headlines1 = soup1.find('body').find_all('h2')
sentences = []
for x in headlines1:
  sentences.append(x.text.strip())
# for x in headlines2:
#   sentences.append(x.text.strip())

def index_sort(list_var):
  lenght = len(list_var)
  list_index = list(range(0, lenght))

  x = list_var
  for i in range(lenght):
    for j in range(lenght):
      if x[list_index[i]] > x[list_index[j]]:
        #swap
        temp = list_index[i]
        list_index[i] = list_index[j]
        list_index[j] = temp
  return list_index


def bot_response(user_input):
    user_input = user_input.lower()
    sentences.append(user_input)  # adds the user input to the sentence list
    bot_response = ''
    cm = CountVectorizer().fit_transform(sentences)
    similarity_scores = cosine_similarity(cm[-1], cm)
    similarity_scores_list = similarity_scores.flatten()
    index = index_sort(similarity_scores_list)
    index = index[1:]
    response_flag = 0

    j = 0
    for i in range(len(index)):
        if similarity_scores_list[index[i]] > 0.0:
            bot_response = bot_response + ' ' + sentences[index[i]]
            response_flag = 1
            j += 1
        if j > 2:
            break

    if response_flag == 0:
        bot_response = bot_response + ' ' + "Don't really have that in my mind, you can visit https://www.business-standard.com to check it out!"

    sentences.remove(user_input)

    return bot_response


def bot_search(request):
    query = request.GET.get('query')

    try:
      ans = bot_response(query)
      return render(request, 'index.html', {'ans': ans, 'query': query})
    except Exception:
        ans = "Sorry!! Could not find anything!! try refreshing the system!"
        return render(request, 'index.html', {'ans': ans, 'query': query})