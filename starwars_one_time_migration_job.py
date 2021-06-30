import sys
import json
import boto3
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
import pyspark.sql.functions as F
from datetime import datetime
from pyspark.sql.types import StringType
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ["JOB_NAME"])
SECRET_NAME = args["secret_name"]
REGION = "us-east-1"

######################################################
# Get Secrets for Target Credentials 
######################################################

client = boto3.client("secretsmanager", region_name = REGION )
get_secret_response = client.get_secret_value(SecretId = SECRET_NAME)
secret_meta_data = json.loads(get_secret_response['SecretString'])

DOCUMENT_DB_HOSTNAME = secret_meta_data['docDB_host']
DOCUMENT_DB_USERNAME = secret_meta_data['docDB_username']
DOCUMENT_DB_PASSWORD = secret_meta_data['docDB_password']
DATABASE_NAME = secret_meta_data['docDB_dbname']
COLLECTION_NAME = secret_meta_data['docDB_member_search_collectionname']
DOCUMENT_DB_PORT = str(secret_meta_data['docDB_port'])
uri = f'mongodb://{DOCUMENT_DB_HOSTNAME}:{DOCUMENT_DB_PORT}'

########################################################


sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

DataSource0 = glueContext.create_dynamic_frame.from_catalog(
    database="starwarsotl", table_name="star_wars_raw"
)

Transform0 = ApplyMapping.apply(frame = DataSource0, mappings = [("sw_char_first_name", "string", "firstName", "string"),
("sw_char_last_name", "string", "lastName", "string"),
("sw_char_age", "string", "age", "string"),                                                                
("sw_char_birthyear", "string", "birth_year", "string"),
("sw_char_birthdate", "string", "birth_date", "string"),
("sw_char_eye_color", "string", "eye_color", "string"),
("sw_char_gender", "string", "gender", "string"),
("sw_char_hair_color", "string", "hair_color", "string"),
("sw_char_species", "string", "species", "string"),
("sw_errorcode", "string", "errorCode", "string"),
("sw_char_dtls_charNm", "string", "charNm", "string"),
("sw_char_dtls_charScndryNm", "string", "charScndryNm", "string"),
("sw_char_dtls_charSpecies", "string", "charSpecies", "string"),
("sw_char_dtls_charGender", "string", "charGender", "string"),
("sw_char_dtls_charMass", "string", "charMass", "string"),
("sw_char_dtls_charHeight", "string", "charHeight", "string"),
("sw_char_dtls_charHomeworld", "string", "charHomeworld", "string"),
("sw_char_dtls_charEyecolor", "string", "charEyecolor", "string"),
("sw_char_dtls_charSkincolor", "string", "charSkincolor", "string"),
("sw_char_dtls_charDied", "string", "charDied", "string"),
("sw_char_height", "string", "height", "string")], transformation_ctx = "Transform0")

from awsglue.dynamicframe import DynamicFrame
starwars_DF = Transform0.toDF()
Transform1  = DynamicFrame.fromDF(starwars_DF, glueContext, "Transform1")
Transform2 = Transform1.toDF()

# Date Conversion Function
def user_defined_timestamp(date_col):
    try:
        if date_col != "0":
            _date = datetime.strptime(date_col, '%Y%m%d')
            return _date.strftime('%Y%m%d')
        else:
            return None
    except:
        return None
user_defined_timestamp_udf = F.udf(user_defined_timestamp, StringType())


# Tranform patBirthDt to date format
Transform2 = Transform2.withColumn("dob", user_defined_timestamp_udf(F.col("dob")))

# Concat firstname and lastname for a new column "name"
Transform2 = Transform2.withColumn("name", F.concat(F.col("firstName"),(F.col("lastName"))))
Transform2 = Transform2.drop("firstName")
Transform2 = Transform2.drop("lastName")

# Transform species
Transform2 = Transform2.withColumn("species", F.when(F.col("species").isNull(), 'Unknown').otherwise(F.col("species")))
 
## Adding derived field clmSttsCd

def derive_homeworld(name):
    if name == "Darth Maul":
        val = "Dathomir"
    elif name == "Jar Jar Binks":
        val = "Naboo"
    elif name == "Darth Sidious":
        val = "Naboo"
    else:
        val = None
    return val

derive_homeworld_udf = F.udf(derive_homeworld, StringType())
Transform2 = Transform2.withColumn("homeworld", derive_homeworld_udf(F.col("name")))


# Transform errorCode to list
errorCode_struct = ArrayType(StringType())

def transform_errorCode(col): 
    if col.endswith(","):
        return col.split(",")[:-1]
    else:
        return col.split(",")

transform_errorCode_udf = F.udf(transform_errorCode, errorCode_struct)
Transform2 = Transform2.withColumn("errorCode", transform_errorCode_udf(F.col("errorCode")))


# Validate name is not null
Transform2 = Transform2.withColumn(
    "val_errors", 
    F.when((F.col("name").isNull()) | (F.col("name") == ""), 
           F.array(F.lit("Name not found!")))
     .otherwise(F.array()))


### Array transformation for charDtls
Transform3 = Transform2.withColumn('charDtls',
                            F.to_json(F.struct("charNm", "charScndryNm", "charSpecies", "charGender", "charMass", "charHeight", 
                                    "charHomeworld", "charEyecolor", "charSkincolor", "charDied")))

json_schema = spark.read.json(Transform3.rdd.map(lambda row: row.charDtls)).schema
Transform3 = Transform3.withColumn('charDtls', F.from_json(F.col('charDtls'), json_schema))

Transform3 = Transform3.drop("charNm", "charScndryNm", "charSpecies", "charGender", "charMass", "charHeight", 
                                    "charHomeworld", "charEyecolor", "charSkincolor", "charDied")     

starwars_dyf = DynamicFrame.fromDF(Transform3, glueContext, "starwars_dyf")

write_documentdb_options = {
    "uri": uri,
    "database": DATABASE_NAME,
    "collection": COLLECTION_NAME,
    "username": DOCUMENT_DB_USERNAME,
    "password": DOCUMENT_DB_PASSWORD
}
datasink = glueContext.write_dynamic_frame.from_options(starwars_dyf, connection_type="documentdb", connection_options=write_documentdb_options)

job.commit()
