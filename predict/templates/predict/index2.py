<h1>EPL 18/19 Prediction Program</h1>
<h2>Latest Registered EPL Match: 18/4/2019</h2>

<div class="selectTeam">
    
    <p><label>Home Team</label></p><br/>
        {% if teamList %}
            <select onchange=filter.submit() name="homeTeam" id="homeTeam">
                {% for team in teamList %}
                <option value="team.name">{{team.name}}</option>
                {% endfor %}
            </select>
            
        {% else %}
        <p>No teams are available.</p>
        {% endif %}
    
    <p><label>Away Team</label></p><br/>
        {% if teamList %}
            <select onchange=filter.submit() name="awayTeam"> id="awayTeam">
            {% for team in teamList %}
            <option value="team.name">{{team.name}}</option>
            {% endfor %}
            </select>
                
        {% else %}
        <p>No teams are available.</p>
        {% endif %}
	
     <form class="chooseTeams" name= "chooseTeams" {% url 'predict:results'%}"method="post">
	{% csrf_token%}
	<input type="submit" name="submit" value="submit"></input>

     </form>
</div>

<head>
    #If you don't want to download and host jQuery yourself, you can include it from a CDN (Content Delivery Network).
    #Use jQuery from Google
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
</head>

