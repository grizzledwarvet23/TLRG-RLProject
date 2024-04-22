using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.SceneManagement;

public class PythonConnector : MonoBehaviour
{
    Thread receiveThread;
    UdpClient client;
    public int port = 5065; // Select a port that fits your setup

    public string pythonIP = "127.0.0.1";

    private PlayerRL agent;

    private int timeStepLength = 10;

    private ConcurrentQueue<Action> mainThreadActions = new ConcurrentQueue<Action>();

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


                if (text.Contains("Action:"))
                {
                    //we will check dat it is water.
                    Debug.Log(text);

                    if(text.Contains("Water:"))
                    {
                        
                        string thirdValueString = text.Substring(text.IndexOf("[") + 1, text.IndexOf("]") - text.IndexOf("[") - 1);
                        int thirdValue = int.Parse(thirdValueString);
                        
                        string rotateValueString = text.Substring(text.IndexOf("[", text.IndexOf("[") + 1) + 1, text.IndexOf("]", text.IndexOf("]") + 1) - text.IndexOf("[", text.IndexOf("[") + 1) - 1);
                        float rotateValue = float.Parse(rotateValueString);
                        mainThreadActions.Enqueue(() => agent.SetAction(0, thirdValue, rotateValue));
                    }
                    else if(text.Contains("Fire:"))
                    {
                        string thirdValueString = text.Substring(text.IndexOf("[") + 1, text.IndexOf("]") - text.IndexOf("[") - 1);
                        int thirdValue = int.Parse(thirdValueString);
                        
                        string rotateValueString = text.Substring(text.IndexOf("[", text.IndexOf("[") + 1) + 1, text.IndexOf("]", text.IndexOf("]") + 1) - text.IndexOf("[", text.IndexOf("[") + 1) - 1);
                        float rotateValue = float.Parse(rotateValueString);
                        mainThreadActions.Enqueue(() => agent.SetAction(1, thirdValue, rotateValue));
                    } 
                    else {
                        Debug.Log("NEITHER!");
                    }
                }

                mainThreadActions.Enqueue(() =>
                {
                    int reward = agent.GetReward();
                    float[] observations = agent.GetObservations();
                    int topDecision = agent.GetTopDecision();

                    //printthe length of observations
                    string json = "{\"PlayerHealth\":[" + observations[0] +
                     "], \"EnemyData\":[" + observations[1] + "," + observations[2] + "," + observations[3] + "," + observations[4] + "," + observations[5] + "," + observations[6] + "," + observations[7] + "," + observations[8] + "," + observations[9] +
                     "], \"CropData\":[" + observations[10] + "," + observations[11] + 
                     "], \"Reward\":" + reward + ", \"TopDecision\":" + topDecision + "}";
                    
                    byte[] sendData = Encoding.UTF8.GetBytes(json);
                    client.Send(sendData, sendData.Length, anyIP);
                });
                
                
            }
            catch (Exception e)
            {
                Debug.Log(e.ToString());
            }
            //sleep for 1 second
            Thread.Sleep(timeStepLength);
        }
    }

    void Update()
    {
        while (mainThreadActions.TryDequeue(out var action))
        {
            action.Invoke();
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
