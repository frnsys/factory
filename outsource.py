import os
import json
import click
import boto3
import base64
import requests
from time import time
from colorama import Fore
from functools import wraps
from collections import defaultdict


def _load_conf():
    path = os.path.expanduser('~/.factory.json')
    conf = json.load(open(path))
    return conf


def connect_ec2(f):
    @wraps(f)
    def func(*args, **kwargs):
        conf = _load_conf()
        sess = boto3.session.Session(profile_name=conf['profile'],
                                     region_name=conf['region'])
        ec2 = sess.resource('ec2')

        return f(ec2, *args, **kwargs)
    return func


@click.group()
def cli():
    pass


@cli.command()
@connect_ec2
@click.argument('tag')
@click.argument('instance_type')
@click.argument('user_data')
def request(ec2, tag, instance_type, user_data):
    conf = _load_conf()
    ami = conf['ami']
    key_name = conf['key_name']

    if os.path.isfile(user_data):
        user_data = open(user_data, 'r').read()
        user_data = base64.b64encode(user_data.encode('utf-8'))
        user_data = user_data.decode('utf-8')

    bid = estimate_spot_price(ec2, instance_types=[instance_type])

    # Create security group
    sg_name = 'spot-sg-{}'.format(tag)

    # Delete it if it exists
    for sg in ec2.security_groups.all():
        if sg.group_name == sg_name:
            click.echo('Found security group {}, deleting...'.format(sg_name))
            sg.delete()
            break

    sec_group = ec2.create_security_group(GroupName=sg_name,
                                          Description='Security group for spot instance request tagged "{}"'.format(tag))

    # Authorize SSH access from the current machine
    this_ip = requests.get('https://api.ipify.org/?format=json').json()['ip']
    sec_group.authorize_ingress(FromPort=22,
                                ToPort=22,
                                CidrIp='{}/32'.format(this_ip),
                                IpProtocol='tcp')
    security_group = sec_group.group_name
    click.echo('Security group "{}" created, allowing SSH from {} (this machine)'.format(security_group, this_ip))

    req = ec2.meta.client.request_spot_instances(SpotPrice=bid,
                                                 LaunchSpecification={
                                                     'ImageId': ami,
                                                     'KeyName': key_name,
                                                     'SecurityGroups': [security_group],
                                                     'UserData': user_data,
                                                     'InstanceType': instance_type,
                                                 })
    req_id = req['SpotInstanceRequests'][0]['SpotInstanceRequestId']
    click.echo('Spot instance requested (request id: {}{}{})...'.format(Fore.BLUE, req_id, Fore.RESET))

    try:
        start = time()
        click.echo('Waiting (this can take a few minutes)...')
        waiter = ec2.meta.client.get_waiter('spot_instance_request_fulfilled')
        waiter.wait(SpotInstanceRequestIds=[req_id])
        end = time()

        req = ec2.meta.client.describe_spot_instance_requests(SpotInstanceRequestIds=[req_id])
        instance_id = req['SpotInstanceRequests'][0]['InstanceId']

        instance = ec2.Instance(instance_id)
        instance.create_tags(Tags=[{
            'Key': 'name',
            'Value': tag
        }])
        click.echo('Instance launched at {}{}{}'.format(Fore.GREEN, instance.public_ip_address, Fore.RESET))
        click.echo('You can ssh into the instance with:')
        click.echo('\tssh -i /path/to/{}.pem ubuntu@{}'.format(key_name, instance.public_ip_address))
        click.echo('Request fulfilled in {:.2f}s'.format(end-start))
    except Exception:
        click.echo('Exception caught. Cleaning up...')
        ec2.meta.client.cancel_spot_instance_requests(SpotInstanceRequestIds=[req_id])
        raise


@cli.command()
@connect_ec2
@click.argument('request_id')
def cancel(ec2, request_id):
    """
    Cancel a spot instance request
    """
    ec2.meta.client.cancel_spot_instance_requests(SpotInstanceRequestIds=[request_id])
    click.echo('Done')


@cli.command()
@connect_ec2
def ls(ec2):
    """
    List open spot instance requests and running spot instances
    """
    res = ec2.meta.client.describe_spot_instance_requests()
    spots = [s for s in res['SpotInstanceRequests'] if s['State'] in ['open', 'active']]
    for spot in spots:
        click.echo('Request id: {}'.format(spot['SpotInstanceRequestId']))
        click.echo('Instance type: {}'.format(spot['LaunchSpecification']['InstanceType']))
        click.echo('State: {}'.format(spot['State']))

        if spot['State'] == 'active':
            id = spot['InstanceId']
            it = ec2.Instance(id)
            click.echo('Instance id: {}'.format(id))
            click.echo('Launched Availability Zone: {}'.format(spot['LaunchedAvailabilityZone']))
            click.echo('Tags: {}'.format(it.tags))
            click.echo('Public IP: {}'.format(it.public_ip_address))
        click.echo('--------\n')


@cli.command()
@connect_ec2
@click.argument('tag')
def terminate(ec2, tag):
    """
    Terminate a spot instance
    """
    instances = ec2.instances.filter(Filters=[{
        'Name': 'tag:name',
        'Values': [tag]
    }])

    if not list(instances):
        click.echo('No instances with tag "{}"!'.format(tag))
        return

    for i in instances:
        click.echo('Terminating {}...'.format(i.id))
        i.terminate()
        i.wait_until_terminated()

    # Check if a security group was created
    sg_name = 'spot-sg-{}'.format(tag)
    sec_groups = ec2.security_groups.filter(GroupNames=[sg_name])
    if sec_groups:
        click.echo('Found security group {}, deleting...'.format(sg_name))
        for sg in sec_groups:
            sg.delete()

    click.echo('Done')


def estimate_spot_price(ec2, instance_types):
    """
    Fetches and presents spot price data
    """
    click.echo('Fetching spot price history...')
    prices = ec2.meta.client.describe_spot_price_history(InstanceTypes=instance_types,
                                                         ProductDescriptions=['Linux/UNIX'])
    prices_by_az = defaultdict(list)
    for p in prices['SpotPriceHistory']:
        az = p['AvailabilityZone']
        prices_by_az[az].append(float(p['SpotPrice']))

    means = []
    for az, ps in prices_by_az.items():
        click.echo('{}{}{}'.format(Fore.BLUE, az, Fore.RESET))
        mean = sum(ps)/len(ps)
        click.echo('\tMean spot price: {}'.format(mean))
        click.echo('\tMin spot price: {}'.format(min(ps)))
        click.echo('\tMax spot price: {}'.format(max(ps)))
        means.append(mean)

    suggested_bid = sum(means)/len(means)
    bid = input('Enter a bid (suggested: {}): '.format(suggested_bid))
    if not bid:
        bid = str(suggested_bid)

    return bid


if __name__ == "__main__":
    cli()
