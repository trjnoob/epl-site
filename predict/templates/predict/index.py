<h1>EPL 18/19 Prediction Program</h1>
<h2>Latest Registered EPL Match: 18/4/2019</h2>

<form action="{% url 'predict:results'%}" method="GET">
    {% csrf_token%}
    {{ form.as_ul }}
    <input type="submit" value="Predict">
</form>
