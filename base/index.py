from django.http import HttpResponse
from django.shortcuts import render

def index(request):
    return render(request,'index.html')
def ITC(request):
    return render(request,'ITC.html')
def adani(request):
    return render(request,'adani.html')
def reliance(request):
    return render(request,'reliance.html')
def bajaj(request):
    return render(request,'bajaj.html')
def HUL(request):
    return render(request,'HUL.html')