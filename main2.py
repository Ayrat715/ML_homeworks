import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

def encode_categorical_features(data):

    # Кодирование категориальных признаков в числовые с помощью LabelEncoder.

    label_encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(exclude=['number']).columns  # Определяем категориальные столбцы

    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column].astype(str))  # Преобразуем каждую категориальную колонку в числовую

    return data

def remove_highly_correlated_features(data, threshold=0.8):

    # Удаление признаков, обладающих высокой взаимной корреляцией (выше заданного порога).

    correlation_matrix = data.corr().abs()  # Вычисляем матрицу корреляций по модулю

    upper_triangle = correlation_matrix.where(  # Сохраняем только верхний треугольник матрицы корреляций
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)  # Создаем маску для верхнего треугольника
    )

    # Находим имена признаков, которые сильно коррелируют друг с другом
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    print("Удаляемые признаки из-за высокой корреляции:")
    for feature in to_drop:
        print(f"- {feature}")

    return data.drop(columns=to_drop)  # Удаляем выбранные признаки из данных

def main():
    # Загрузка данных
    data = pd.read_csv("/Users/ajratfahrutdinov/PycharmProjects/ML_second_homework/AmesHousing.csv")

    # Деление на числовые и категориальные признаки
    numeric_data = data.select_dtypes(include=['number'])
    categorical_data = data.select_dtypes(exclude=['number'])

    # Кодирование категориальных признаков
    categorical_data_encoded = encode_categorical_features(categorical_data)

    # Объединение обратно в один датафрейм
    data = pd.concat([numeric_data, categorical_data_encoded], axis=1)

    # Удаление сильно коррелирующих признаков
    data = remove_highly_correlated_features(data, threshold=0.8)

    # Визуализация корреляционной матрицы после удаления признаков
    plt.figure(figsize=(48, 40))  # Устанавливаем большой размер графика
    sns.heatmap(data=data.corr(), annot=True, cmap="coolwarm", fmt=".2f")  # Строим тепловую карту корреляций
    plt.show()

    # Деление на признаки и целевую переменную
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]

    # Заполнение пропусков средним значением
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Снижение размерности признаков с помощью PCA до 2 компонентов
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    # Создание 3D-графика зависимости двух главных компонент и целевой переменной
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        X_pca[:, 0],  # Координата x — первая главная компонента
        X_pca[:, 1],  # Координата y — вторая главная компонента
        y,            # Координата z — целевая переменная SalePrice
        c=y,          # Цвет точек отражает значение SalePrice
        cmap="viridis"
    )

    ax.set_xlabel("Главная компонента 1 (PCA)")
    ax.set_ylabel("Главная компонента 2 (PCA)")
    ax.set_zlabel("Целевое значение (SalePrice)")

    # Добавление цветовой шкалы
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("SalePrice")

    plt.title("3D-график данных после PCA")
    plt.show()

    # Разделение данных на обучающую и тестовую выборки
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Задаем диапазон значений параметра регуляризации alpha
    alphas = np.logspace(-3, 2, 50)  # Логарифмически равномерные значения от 0.001 до 100

    # Список для хранения ошибок модели на тестовой выборке
    lasso_errors = []

    # Перебор всех значений alpha для поиска оптимального
    for alpha in alphas:
        lasso_model = Lasso(alpha=alpha, max_iter=10000)
        lasso_model.fit(x_train, y_train)

        y_pred_lasso = lasso_model.predict(x_test)

        rmse = mean_squared_error(y_test, y_pred_lasso)  # Вычисляем среднеквадратичную ошибку
        lasso_errors.append(rmse)

    # Нахождение оптимального значения alpha (с минимальной ошибкой)
    optimal_alpha = alphas[np.argmin(lasso_errors)]
    print(f"Оптимальное значение alpha: {optimal_alpha}")

    # Визуализация зависимости ошибки от alpha
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, lasso_errors, label="Lasso", color="blue")
    plt.xscale("log")  # Логарифмическая шкала по оси x
    plt.xlabel("Alpha (коэффициент регуляризации)")
    plt.ylabel("RMSE (среднеквадратичная ошибка)")
    plt.title("Зависимость ошибки от коэффициента регуляризации (Lasso)")
    plt.axvline(optimal_alpha, color="red", linestyle="--", label=f"Оптимальное alpha: {optimal_alpha:.3f}")
    plt.legend()
    plt.show()

    # Обучение модели Lasso с оптимальным значением alpha
    lasso_model = Lasso(alpha=optimal_alpha, max_iter=10000)
    lasso_model.fit(x_train, y_train)

    # Извлечение коэффициентов обученной модели
    feature_names = data.drop(columns=["SalePrice"]).columns  # Названия признаков
    coefficients = lasso_model.coef_  # Коэффициенты модели

    # Формируем датафрейм для анализа важности признаков
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    # Добавляем столбец с абсолютным значением коэффициента для сортировки
    coef_df['Absolute Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Absolute Coefficient', ascending=False)

    # Вывод наиболее влиятельного признака
    most_influential_feature = coef_df.iloc[0]
    print(f"Наиболее влиятельный признак: {most_influential_feature['Feature']}")
    print(f"Коэффициент: {most_influential_feature['Coefficient']}")

    # Визуализация топ-10 наиболее влиятельных признаков
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Absolute Coefficient', y='Feature', data=coef_df.head(10))
    plt.title("Топ-10 наиболее влиятельных признаков")
    plt.show()

if __name__ == "__main__":
    main()
