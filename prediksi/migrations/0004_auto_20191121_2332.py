# Generated by Django 2.2.5 on 2019-11-21 16:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prediksi', '0003_resultstocks'),
    ]

    operations = [
        migrations.DeleteModel(
            name='ResultStocks',
        ),
        migrations.AddField(
            model_name='stocks',
            name='grouped1',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='stocks',
            name='grouped20',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='stocks',
            name='grouped5',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='stocks',
            name='onehot1',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='stocks',
            name='onehot20',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='stocks',
            name='onehot5',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='stocks',
            name='plain1',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='stocks',
            name='plain20',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='stocks',
            name='plain5',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
    ]