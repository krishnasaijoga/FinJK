
import {
  AgentCoreApplication,
  type AgentCoreApplicationProps
} from '@aws/agentcore-cdk';

import { Stack, type StackProps } from 'aws-cdk-lib';
// import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';

export interface AgentCoreStackProps extends StackProps {
  application: AgentCoreApplicationProps;
}

export class AgentCoreStack extends Stack {
  public readonly application: AgentCoreApplication;

  constructor(scope: Construct, id: string, props: AgentCoreStackProps) {
    super(scope, id, props);

    // Create AgentCore resources from agentcore.json
    this.application = new AgentCoreApplication(
      this,
      'Application',
      props.application
    );

    // Reference the existing FinJK Secrets Manager secret

    // Release below filters before deploying on AgentCore

    // const finjkSecret = secretsmanager.Secret.fromSecretCompleteArn(
    //   this,
    //   'FinJKAPIKeys',
    //   'arn:aws:secretsmanager:ap-southeast-2:580064883885:secret:finjk/api-keys-YEKJKa'
    // );

    // // Access the runtime named "agent"
    // const agentRuntime = this.application.runtime('agent');

    // // Grant the runtime permission to retrieve the secret
    // finjkSecret.grantRead(agentRuntime);

    // // Pass the secret identifier, not its value
    // agentRuntime.addEnvironmentVariable(
    //   'FINJK_SECRET_ID',
    //   finjkSecret.secretArn
    // );
  }
}
