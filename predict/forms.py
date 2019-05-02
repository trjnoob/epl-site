from django import forms
from .models import Team
from predict import predict_nn2

#teamList = Team.objects.order_by('name')
teamList = predict_nn2.getTeamList(19)
currentSeasonTeams=[(x,x) for x in teamList]
currentSeasonTeams.sort()

class TeamForm(forms.Form):
    homeTeam = forms.CharField(label="Home Team",
                               widget=forms.Select(choices=currentSeasonTeams))

    awayTeam = forms.CharField(label="Away Team",
                               widget=forms.Select(choices=currentSeasonTeams))
    '''
     def cleanData(self):
         homeData = self.cleaned_data['homeTeam']
         awayData = self.cleaned_data['awayTeam']

         return homeData, awayData
        '''
