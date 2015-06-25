# Factory

Build various NLP models in a parallel manner


## Usage

For more efficient memory-usage in parallel processing, run the bigram phrase model as a separate process:

    $ python service.py phrases

The factory will automatically connect to this process if it is needed and available. Otherwise, it falls back to loading the phrases model directly.


Then run the command you want, for example:

    $ python train.py train_idf "data/*.txt" data/idf.json method=word

You can see all available commands by running:

    $ python train.py

The structure of the commands are generally:

    $ python train.py <command> <path to input> <path to ouput> <kwargs>


## AWS Spot Instances

On consumer hardware, these models can sometimes take a very long time to train. Parallelization doesn't always help since it may increase memory requirements to beyond what's available on your own machine.

`factory` can take advantage of the cheap pricing of AWS spot instances. You can easily bid on short-term computing power, quickly train your model (e.g. on a GPU or memory optimized instance), and download the resulting model.

Specify your AWS credentials in `~/.aws/credentials`:

    [default]
    aws_access_key_id = YOURACCESSKEY
    aws_secret_access_key = YOURSECRETKEY

    [another_profile]
    aws_access_key_id = YOURACCESSKEY
    aws_secret_access_key = YOURSECRETKEY

Then create a `factory` config at `~/.factory.json` with the following (filled with whatever values you prefer, of course):

    {
        "region": "us-east-1",
        "profile": "another_profile",
        "ami": "ami-a0c499c8",
        "key_name": "my_key_pair"
    }

If it will be consistent across your usage.

Then you have a few options:

    # Create a spot instance request
    $ python outsource.py request <name> <instance type> [--user_data=<path to user data script>] [--ami=<image id>]
    $ python outsource.py request doc2vec r3.2xlarge --user_data=user_data.sh
    # Then you can do:
    $ ssh -i my_key_pair.pem ubuntu@<instance's public ip>
    # Try `ec2-user` if `ubuntu` doesn't work

    # Cancel a spot instance request
    $ python outsource.py cancel <request id>
    $ python outsource.py cancel sir-02baen2k

    # Terminate a spot instance
    $ python outsource.py terminate <name>
    $ python outsource.py terminate doc2vec

    # List spot instances and requests
    $ python outsource.py ls

What `factory` does for you:
- it will look at current bids for your requests instance type in your region, list out their min/max/mean, and suggest a bid price.
- it will create a security group with SSH access for your local machine (i.e. the machine making the spot instance request). It will also clean up this security group when you terminate the instance.
- it will tell you the public IP of the spot instance when it's ready.

Reference:

- Ubuntu AMIs: <https://cloud-images.ubuntu.com/releases/>
- Spot instance types: <https://aws.amazon.com/ec2/purchasing-options/spot-instances/>
- If your user data script doesn't seem to be executing...
    - check that your script is at `/var/lib/cloud/instance/scripts/part-001`
    - check the logs at `/var/log/cloud-init.log`
    - check the logs at `/var/log/cloud-init-output.log`
