from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse, HttpResponseRedirect
from .models import Team
from django.urls import reverse
from .forms import TeamForm
from predict import predict_nn2
import numpy as np
import matplotlib.pyplot as plt, pylab
import PIL, PIL.Image, io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from base64 import b64encode

# Create your views here.

'''
def heading(request):
    return HttpResponse("EPL 18/19 Prediction Program")
'''
'''
def process(request):
    homeTeam = get_object_or_404(Team, name="homeTeam")
    awayTeam = get_object_or_404(Team, name="awayTeam")
    
    if request.method == 'GET':
        form = TeamForm(request.GET)
        if form.is_valid():
            homeTeam = form.cleaned_data['homeTeam']
            awayTeam = form.cleaned_data['awayTeam']

            return HttpResponseRedirect(reverse('predict:results'))

        context = {
            'form' : form,
            'homeTeam' : homeTeam,
            'awayTeam' : awayTeam,
        }
        return(render(request, 'predict/index.html', context))
'''


'''
def index(request):
    teamList = Team.objects.order_by('name')

    if request.method == 'GET':
        form = TeamForm(request.GET)
    
    context = {
            'form' : form,
            'Team' : teamList,
        }
    return(render(request, 'predict/index.html', context))
'''

def index(request):
    teamList = predict_nn2.getTeamList(19)

    if request.method == 'GET':
        form = TeamForm(request.GET)
    
    context = {
            'form' : form,
            'teamList' : teamList,
        }
    return(render(request, 'predict/index.html', context))
    

def results(request):
    if request.method == 'GET':
        form = TeamForm(request.GET)
        if form.is_valid():
            homeTeam = form.cleaned_data['homeTeam']
            awayTeam = form.cleaned_data['awayTeam']

    #Determine the probability of win, draw and lose for the selected teams
    teamList = predict_nn2.getTeamList(19)
    probList = predict_nn2.predict_match(19, teamList.index(homeTeam), teamList.index(awayTeam))
    
    context = {
            'form' : form,
            'homeTeam' : homeTeam,
            'awayTeam' : awayTeam,
            'probList' : probList,  
    }
    
    
    #Plot Graph 
    x = np.arange(3)
    a = probList
    y = np.squeeze(np.asarray(a)) * 100
    #y = [y[0][0], y[1][0], y[2][0]]
    print(y)
    plt.title(homeTeam + " (H)\n VS \n" + awayTeam + ' (A)') 
    plt.bar(x, y, width = 0.4)
    for a,b in zip(x, y):
        plt.text(a, b, str(round(b, 1)) + '%', horizontalalignment='center')
    plt.xticks(x, [homeTeam + ' Win', 'Draw', awayTeam + ' Win'])
    plt.ylabel("Probability(%)")
    #image = io.StringIO()
    #plt.savefig(image, format='png')
    #pred64 = base64.b64encode(image.read())
    #context['pred64'] = pred64
    #plt.savefig('predict/templates/predict/pred.png')
    #plt.show()

    '''
    fig = Figure()
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    #return response
    '''
    
    '''
    # Store image in a string buffer
    buffer = io.StringIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    '''

    return(render(request, 'predict/results.html', context))



