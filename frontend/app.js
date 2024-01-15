async function predictWeather() {
    const temperature = parseFloat(document.getElementById("temperature").value);
    const feelslike = parseFloat(document.getElementById("feelslike").value);
    const dewpoint = parseFloat(document.getElementById("dewpoint").value);
    const humidity = parseFloat(document.getElementById("humidity").value);
    const precipitation = parseFloat(document.getElementById("precipitation").value);
    const precipprob = parseFloat(document.getElementById("precipprob").value);
    const preciptype = document.getElementById("preciptype").value;
    const snow = parseFloat(document.getElementById("snow").value);
    const snowdepth = parseFloat(document.getElementById("snowdepth").value);
    const windgust = parseFloat(document.getElementById("windgust").value);
    const windspeed = parseFloat(document.getElementById("windspeed").value);
    const winddirection = parseFloat(document.getElementById("winddirection").value);
    const sealevelpressure = parseFloat(document.getElementById("sealevelpressure").value);
    const cloudcover = parseFloat(document.getElementById("cloudcover").value);
    const visibility = parseFloat(document.getElementById("visibility").value);
    const solarradiation = parseFloat(document.getElementById("solarradiation").value);
    const uvindex = parseFloat(document.getElementById("uvindex").value);
    const severerisk = parseFloat(document.getElementById("severerisk").value);
    const conditions = document.getElementById("conditions").value;

    const weatherData = {
        temperature,
        feelslike,
        dewpoint,
        humidity,
        precipitation,
        precipprob,
        preciptype,
        snow,
        snowdepth,
        windgust,
        windspeed,
        winddirection,
        sealevelpressure,
        cloudcover,
        visibility,
        solarradiation,
        uvindex,
        severerisk,
        conditions,
    };

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(weatherData),
        });

        if (!response.ok) {
            throw new Error('Failed to get prediction.');
        }

        const result = await response.json();
        document.getElementById("result").innerHTML = `<p>Demand: ${result.demand}, Price: ${result.price}</p>`;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById("result").innerHTML = '<p>Failed to get prediction. Please check your input.</p>';
    }
}
