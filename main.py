import numpy
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
print(df)

# 20 Task
# 1.	Sürətli baxış:
# ○	head(5), tail(5) və sample(3) ilə datasetə bax.
print(df.head(5))
print(df.tail(5))
print(df.sample(3))
# ○	Sual: Səncə hansı 3 sətir “maraqlı” görünür və niyə?
# Answer:
# 1.Sample() methodu her defesinde random row return etdi
# 2.141m2 1 otag  Binagadi 1 otagli 141m2 maragli ola biler
print(df[(df['Area_m2'] == 141) & (df['District'] == 'Binagadi')])
print(df.query("Area_m2 == 141 and District == 'Binagadi'"))  ##method2
# 3.111m2 2 otag  Nizami  111 kvadrata 2 otag azdir.
print(df.query('Area_m2 == 111 and District == "Nizami"'))

# 2.	Struktur yoxlaması:
# ○	info() nəticəsinə əsasən hansı sütunlarda boş dəyər var?
# ○	Hər sütunun dtype-ını qeyd et.
print(df.info())
print(df.dtypes)
# 3.	Statistik icmal:
#with function
def summary_stats(df_params, column):
    mean_val = df_params[column].mean()
    median_val = df_params[column].median()
    std_val = df_params[column].std()

    print(f"Column: {column}")
    print(f"Mean: {mean_val}")
    print(f"Median: {median_val}")
    print(f"Std: {std_val}")


# ○	describe() nəticəsinə bax və Area_m2, Price_AZN üçün mean, median, std dəyərlərini müqayisə et.
print(df.describe())
summary_stats(df, 'Area_m2')
summary_stats(df, 'Price_AZN')
# ○	Yekun: Price paylanması simmetrikdirmi?
#DEyil
# 4.	Tip düzəlişi:

# ○	Price_AZN-də string dəyər olub-olmadığını yoxla (var!). Bunu rəqəmə çevir (error='coerce' istifadə edə bilərsən).
print(df['Price_AZN'].apply(type).value_counts())##All Types in price_azn
df['Price_AZN'] = pd.to_numeric(df['Price_AZN'], errors='coerce')
print("Count Of NAN",df['Price_AZN'].isna().sum())
print(df.dtypes)

# 5.	Qiymət outlier-ləri (təxmini):
# ○	Price_AZN-i sortla (azalan). İlk 10 sətirdə outlier təsiri verən hansı dəyərləri görürsən?
sorted_price = df.sort_values(by=['Price_AZN'], ascending=False)
print(sorted_price.head(10)[['Price_AZN', 'Area_m2','District','Rooms']])
print(sorted_price['Price_AZN'])
# ○	“Ən bahalı 3 m²” ideyasını qeyd et (hələ hesablamaya ehtiyac yoxdur).
df['ppm'] = df['Price_AZN'] / df['Area_m2']
df.sort_values(by='ppm', ascending=False).head(3)
# 6.	Kateqorik balans:

# ○	District üçün value_counts() çıxar.
# ○	Sual: Hansi rayon(lar) çox/az təmsil olunub? Bu imbalance nə ola bilər?
print(df['District'].value_counts())
print(df['District'].value_counts(normalize=True)) #Nisbet 0:20
normalized = df['District'].value_counts(normalize=True)
print(normalized.tail(3)) #bottom 3
df['District'].value_counts().plot(kind='bar', title='District Count') ##Chart
plt.show()

#
# 7.	Rooms distribusiyası:
#
# ○	Rooms üçün value_counts().sort_index() çıxar.
rooms_counts = df['Rooms'].value_counts().sort_index()
print(rooms_counts)
print("\n")
for room, count in rooms_counts.items():
    print(f"{room} rooms : {count} home")
# ○	Sual: 1, 2, 3 otaqlılarda paylanma necədir?
plt.figure(figsize=(8,5))
rooms_counts.plot(kind='bar', color='skyblue')
plt.title("Rooms ")
plt.xlabel("Room Count")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# 8.	Mean vs Median (Price):
#
# ○	Price_AZN üçün mean və median müqayisə et.
mean = df['Price_AZN'].mean()
median = df['Price_AZN'].median()
std_price = df['Price_AZN'].std()
print("Mean ----- ",mean,median)
lower_limit = mean - 3 * std_price
upper_limit = mean + 3 * std_price
outliers = df[(df['Price_AZN'] < lower_limit) | (df['Price_AZN'] > upper_limit)]
print("Outlier:")
print(outliers[['District','Rooms','Area_m2','Price_AZN']])
# ○	Fikir: Niyə fərq var? Outlier-lərin rolu nədir?
#Ferq ona gore varki, median pricelari aratan sira ile sort edir be len cutduse ortaki 2 ededin ortasinin qaytariri
#Mean : Ededi ortani tapir yeni sum(df[Price_AZN])/len([Price_AZN]) yeni cemi elemet sayina bolur
#Ikiside Riyazi anlayisdi.
#Oulier ise en uc deyer yeni normadan kenar mushahide olunan deyerdi
#Outlier — datasetdeki diger deyerlerde xeyli ferqlenen ddeyerdi
#[1000,2000,2800,39000,90000000] sonuncu outlierdir


# 9.	Mode və yayılma ölçüləri:
# ○	Rooms üçün mode (ən çox görünən), Price_AZN üçün variance və std hesabla.
# ○	Yekun: Hansı rayonun qiymətlərində yayılma daha çox ola bilər (hipotez)?
rooms_mode = df['Rooms'].mode()[0]
print("Rooms Mode:", rooms_mode)
price_var = df['Price_AZN'].var()
price_std = df['Price_AZN'].std()

print("Price_AZN var:", price_var)
print("Price_AZN std:", price_std)

district_std = df.groupby('District')['Price_AZN'].std()
print(district_std.sort_values(ascending=False))


# 10.	Filter + seçim:
#
# ○	Rooms >= 3 və Area_m2 >= 100 olan sətirləri seç. Bu alt-kəsikdə Price_AZN orta qiyməti neçədir?

condition_apartments = (df[(df['Rooms'] >= 3) & (df['Area_m2']>=100)])
average_price = numpy.mean(condition_apartments['Price_AZN'])
# average_price = condition_apartments['Price_AZN'].mean() ##same method but different modals
print(condition_apartments[['District', 'Rooms', 'Area_m2', 'Price_AZN']])
print(condition_apartments,average_price)

# 11.	District üzrə mərkəz ölçüləri:
#
# ○	groupby("District")["Price_AZN"].agg(["mean","median","count"]) hesabla.
#
# ○	Sual: Harada median mean-dən xeyli fərqlənir və niyə?
#
# 12.	Outlier aşkarlanması (IQR):
#
# ○	Price_AZN üçün Q1, Q3, IQR, lower/upper bound hesabla və “çıxışda” qalan sətirləri göstər.
#
# ○	Qeyd: Bunları avtomatik filtr kimi tətbiq et.
#
# 13.	Outlier aşkarlanması (Z-score):
#
# ○	zscore(Price_AZN) hesabla və |z|>3 sətirləri tap.
#
# ○	Nəticə: IQR və Z-score nəticələri eyni sətirləri göstərirmi?
#
# 14.	Top 10 ən bahalı və ən ucuz evlər:
sorted_home_by_price = df.sort_values(by='Price_AZN', ascending=False)
print('Top 10:',sorted_home_by_price[['District','Price_AZN','Rooms']].head(10))
sorted_home_by_price_min = df.sort_values(by='Price_AZN', ascending=True)
print('Bottom 10:',sorted_home_by_price[['District','Price_AZN','Rooms']].tail(10))
mean_price = df['Price_AZN'].mean()
std_price = df['Price_AZN'].std()
lower_limit = mean_price - 3*std_price
upper_limit = mean_price + 3*std_price
# ○	Qeyd: Outlier-ləri ayrıca qeyd et (əlavə sütun “IsOutlier” ola bilər).
df['IsOutlier'] = numpy.where((df['Price_AZN'] < lower_limit) | (df['Price_AZN'] > upper_limit), True, False)

# 15.	Room-Effect ideyası:
# ○	Rooms ilə Price_AZN arasında “orta qiymətə təsir”i hiss etmək üçün groupby("Rooms")["Price_AZN"].median() çıxar.
room_price_median  = df.groupby('Rooms')['Price_AZN'].median().sort_index()
print('Median Price by Room count:',room_price_median)
# ○	Qeyd: Median niyə daha məntiqli ola bilər?
#Outlierdan tesirlenmir
#Yeni Qiymet paylanmasi std qeyri simetrikdise , median data duz stabil deyer gosterecek
# Mean (ededi orta) uc qiymetlerden (outlier)-dan chox tesirlenir yeni (10000000,999,700,500)/len(num)
# Median ise yalnizca ortadaki deyeri goturduyu uchun bele tesirlerden kenardi.


# 16.	Price per m² (ppm):
#
# ○	Yeni sütun: ppm = Price_AZN / Area_m2 (təhlükəsiz bölmə və boş dəyərləri nəzərə al!).
#
# ○	ppm-ə görə ilk 10 sətiri çıxar. Sual: Hansı rayon önə çıxır?

df['ppm'] = df.apply(lambda x: x['Price_AZN'] / x['Area_m2'] if x['Area_m2'] > 0 else None, axis=1)
print('Top 10:--','\n')
top_ppm = df.sort_values(by='ppm', ascending=False).head(10)
print(top_ppm[['District', 'Rooms', 'Area_m2', 'Price_AZN', 'ppm']])

# 17.	Kateqorik təmizləmə (map):
#
# ○	District-ləri region map ilə qrupla (məs: Sabayil=“Prime”, Yasamal/Nizami/Nasimi/Nerimanov=“Central”, Khatai/Binagadi=“Outer”).
#
# ○	groupby("region")["Price_AZN"].median() müqayisə et.

region_map = {
    'Sabayil': 'Prime',
    'Yasamal': 'Central',
    'Nizami': 'Central',
    'Nasimi': 'Central',
    'Nerimanov': 'Central',
    'Khatai': 'Outer',
    'Binagadi': 'Outer'
}

df['Region'] = df['District'].map(region_map)

region_median = df.groupby('Region')['Price_AZN'].median().sort_values(ascending=False)
print("Median Price by Region:")
print(region_median)


# 18.	Tip problemləri və boşluqların təsiri:
#
# ○	Price_AZN-də boş/NaN olan sətirləri tap; bunların District/Rooms/Area paylanmasını təhlil et.
#
# ○	Qeyd: Boş dəyərləri necə imputasiya edərdin (niyə median daha yaxşıdır)?
missing_price = df[df['Price_AZN'].isna()]
print("Rowda Price_AZN NAN Olanlar:")
print(missing_price[['District','Rooms','Area_m2','Price_AZN']])
print("District none Price_AZN Count:")
print(missing_price['District'].value_counts())

print("\nRooms None Price_AZN Count:")
print(missing_price['Rooms'].value_counts())

print("\nArea_m2 none Price_AZN Std:")
print(missing_price['Area_m2'].describe())
df['Price_AZN'] = df['Price_AZN'].fillna(df['Price_AZN'].median())


# 19.	Simulyasiya “təmiz” qiymət medianı:
# ○	Outlier-ləri IQR ilə filtr edib təmiz subset üçün Price_AZN medianını hesabla.
# ○	Təmiz medianı ümumi medianla müqayisə et.
Q1 = df['Price_AZN'].quantile(0.25)
Q3 = df['Price_AZN'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df[(df['Price_AZN'] >= lower_bound) & (df['Price_AZN'] <= upper_bound)]
overall_median = df['Price_AZN'].median()

clean_median = df_clean['Price_AZN'].median()

print(f"Overall median: {overall_median}")
print(f"Clean median (without outliers): {clean_median}")

#clean daha stabil deyer verdi
#
# 20.	Kiçik “mini-profil” hesabatı yaz:
#
# ○	shape, nulls per column, numeric describe, District count, mean/median Price, top-ppm 5 rows.
#
# ○	5 sətirlik nəticə şərhi əlavə et: “Nə gördün? Nələr risk/ fürsət yaradır?”
