using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class PythonConnector : MonoBehaviour
{
    Thread receiveThread;
    UdpClient client;
    public int port = 5065; // Select a port that fits your setup

    public string pythonIP = "127.0.0.1";

    private PlayerRL agent;

    private int timeStepLength = 100;

    void Start()
    {
        agent = GetComponent<PlayerRL>();
        InitializeUDP();
    }

    private void InitializeUDP()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    private void ReceiveData()
    {
        client = new UdpClient(port);
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Parse("127.0.0.1"), port);
                byte[] data = client.Receive(ref anyIP);

                string text = Encoding.UTF8.GetString(data);

                //if text has Action: , 
                //then we know it is in the format "Action: [<float_value>]".
                //extract the float value and call agent.SetAction(float_value)
                if (text.Contains("Action:"))
                {
                    string actionValueString = text.Substring(text.IndexOf("[") + 1, text.IndexOf("]") - text.IndexOf("[") - 1);
                    float actionValue = float.Parse(actionValueString);
                    agent.SetAction(actionValue);
                }

                //こんにちは、これは日本語のコメントです。お元気ですか、皆さん？僕の名前はザヤーンですよ。よろしくお願いします。
                
                
                // string message = "Hello from Unity!";
                // data = Encoding.UTF8.GetBytes(message);
                
                
                float[] observations = agent.GetObservations();
                int reward = agent.GetReward(); 

                //the way we'll structure the json is an array with two elements.
                //the first element is a dictionary of the state features, ie: {ClosestEnemyPosition: [1.2, 3.2], etc}
                //the second element is just a single number, the reward
                string json = "{\"ClosestEnemyPosition\":[" + string.Join(",", observations) + "], \"Reward\":" + reward + "}";
                data = Encoding.UTF8.GetBytes(json);
                client.Send(data, data.Length, anyIP);



        
                
            }
            catch (Exception e)
            {
                Debug.Log(e.ToString());
            }
            //sleep for 1 second
            Thread.Sleep(timeStepLength);
        }
    }


    

    void OnApplicationQuit()
    {
        if (receiveThread != null) receiveThread.Abort();
        //send the client "quit"
        byte[] data = Encoding.UTF8.GetBytes("quit");
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Parse("127.0.0.1"), port);
        client.Send(data, data.Length, anyIP);
        client.Close();
    }
}