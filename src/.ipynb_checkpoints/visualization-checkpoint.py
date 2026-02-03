import matplotlib.pyplot as plt
import seaborn as sns

def visualization(df):
    # Loan_Status plots
    if 'Loan_Status' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        loan_status_counts = df['Loan_Status'].value_counts()
        total = loan_status_counts.sum()
        loan_status_percentages = (loan_status_counts / total) * 100

        loan_status_counts.plot.bar(ax=axes[0], color=['skyblue', 'lightcoral'], edgecolor='black')
        axes[0].set_title('Distribution of Loan Status (Y/N) - Bar Chart')
        axes[0].set_xlabel('Loan Status')
        axes[0].set_ylabel('Count')

        axes[1].pie(
            loan_status_percentages,
            labels=loan_status_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=['skyblue', 'lightcoral'],
            wedgeprops={'edgecolor': 'black'}
        )
        axes[1].set_title('Distribution of Loan Status (Y/N) - Pie Chart')

        plt.tight_layout()
        plt.show()

    # Numerical columns
    numerical_columns = [col for col in ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome'] if col in df.columns]
    for column in numerical_columns:
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.subplot(1, 2, 2)
        df[column].plot.box()
        plt.title(f'Boxplot of {column}')
        plt.show()

    # Categorical features
    categorical_features = [col for col in ['Gender', 'Married', 'Dependents', 'Education', 
                                            'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Amount_Term'] if col in df.columns]
    plt.figure(figsize=(15, 12))
    for i, column in enumerate(categorical_features, 1):
        plt.subplot(3, 3, i)
        unique_values = df[column].nunique()
        palette = sns.color_palette("Set2", unique_values)
        ax = sns.countplot(data=df, x=column, hue=column, palette=palette, dodge=False)
        plt.title(f'Distribution of {column}')

        legend = ax.get_legend()
        if legend:
            legend.remove()

        total = len(df)
        for p in ax.patches:
            count = p.get_height()
            percentage = 100 * count / total
            if percentage > 0:
                ax.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width()/2, p.get_height()), 
                            ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Target relationship plots
    if 'Loan_Status' in df.columns:
        palette = sns.color_palette("Set2")[:2]
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(categorical_features, 1):
            plt.subplot(3, 3, i)
            sns.countplot(data=df, x=column, hue='Loan_Status', palette=palette)
            plt.title(f'{column} vs Target')
        plt.tight_layout()
        plt.show()

    
