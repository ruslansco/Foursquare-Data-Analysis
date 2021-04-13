# Analysis of Fousquare Check-Ins in New York City and Tokyo.


1) **Preparation of the Datasets** (dataset_TSMC2014_NYC.csv, dataset_TSMC2014_TKY.csv)
![image1](https://user-images.githubusercontent.com/27319299/114584742-cc521d80-9c37-11eb-8efe-d7d8b716eae4.png)

2) **Converted string type values of column - “utcTimestamp” to datetime64[ns] type and disassembled into columns: “Year”, ”Month”, ”Hour”, ”Minute”.**

![image2](https://user-images.githubusercontent.com/27319299/114585202-408cc100-9c38-11eb-9aa9-4f8ed229450e.png)

3) **Mapped latitude and longitude with precision 5, and inserted into new column - “Geohash”, by using pygeohash module. It divided the coordinates into “buckets” of different zones based on number of digits (precision).**

![image3](https://user-images.githubusercontent.com/27319299/114585235-48e4fc00-9c38-11eb-99d2-85c5abc4430c.png)
![image4](https://user-images.githubusercontent.com/27319299/114585275-569a8180-9c38-11eb-905e-836ac358410b.png)

4) **Encoding data types**
Last step of data wrangling before I begin the model evaluation, was encoding Binary ID’s and string values to integer type, by using Label Encoder module. Specifically target value – Venue Category and feature – Geohash.

![image12](https://user-images.githubusercontent.com/27319299/114585936-fe17b400-9c38-11eb-9b27-b58e2b728520.png)
![image13](https://user-images.githubusercontent.com/27319299/114585975-08d24900-9c39-11eb-8940-ecbdf1773212.png)
![image14](https://user-images.githubusercontent.com/27319299/114585976-096adf80-9c39-11eb-9d72-d5fd63795466.png)
![image15](https://user-images.githubusercontent.com/27319299/114585990-0bcd3980-9c39-11eb-8c4d-da879cf63cbe.png)
![image16](https://user-images.githubusercontent.com/27319299/114586010-125bb100-9c39-11eb-8110-5a3174a61e7b.png)
![image17](https://user-images.githubusercontent.com/27319299/114586150-34edca00-9c39-11eb-8b69-cb2f418b30db.png)
![image18](https://user-images.githubusercontent.com/27319299/114586154-35866080-9c39-11eb-809d-2c0c3fab8dc6.png)

5) **Target Value: Venue Category. 	Features: Year, Month, Weekday, Hour, Minute, and Geohash.** 
![image19](https://user-images.githubusercontent.com/27319299/114586174-3a4b1480-9c39-11eb-8067-740487b44538.png)

6) **Datasets Validations **
![image19](https://user-images.githubusercontent.com/27319299/114586272-564eb600-9c39-11eb-95ff-4cee5f3ad850.png)

# Files
**foursquare_checkins_scatters.py**
![Check-Ins NYC](https://user-images.githubusercontent.com/27319299/114583756-c1e35400-9c36-11eb-8857-30d34633b49a.PNG)

![Check-Ins Tokyo](https://user-images.githubusercontent.com/27319299/114583715-b98b1900-9c36-11eb-93ec-06d059257d1c.PNG)

**popular_venues_charts.py**
![10 Popular Venues - NYC](https://user-images.githubusercontent.com/27319299/114584110-2acacc00-9c37-11eb-8c88-2b504b940135.PNG)
![10 Popular Venues - Tokyo](https://user-images.githubusercontent.com/27319299/114584115-2bfbf900-9c37-11eb-9efd-863f630b7c1f.PNG)

**evaluate_best_model.py**
![Precision 5, venueCategoryId(feature)](https://user-images.githubusercontent.com/27319299/114586450-87c78180-9c39-11eb-8cfa-7c3fa6ed02ba.PNG)
