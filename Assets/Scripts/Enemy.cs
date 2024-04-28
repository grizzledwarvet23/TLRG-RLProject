using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Enemy : MonoBehaviour
{
    public float movementSpeed;
    public float health;
    private float maxHealth;
    GameObject player;
    Rigidbody2D rb;
    public Image healthBar;
    public Canvas canvas;
    [System.NonSerialized]
    public bool isSlowed;
    [System.NonSerialized]
    public bool isBurning;
    float timeLastBurnedDamage;

    bool touchingRock;
    GameObject touchedRock;
    float timeLastDamagedRock;
    float rockDamageDelay = 1.2f; //attacking the rock
    float fireDamageDelay = 0.75f; //enemy getting burned
    // Start is called before the first frame update
    public ParticleSystem wetParticles;

    public GameObject chainLightningPrefab;

    [System.NonSerialized]
    public int hitByWaterCount = 0;
    [System.NonSerialized]
    public float timeLastWaterHit = 0f;

    public AudioSource burnSound;

    bool nearPlayer = false;
    float attackRate = 1f;
    float timeLastAttacked = 0f;
    public int damage;

    float timePenalty = 0f;

    void Start()
    {
        player = GameObject.Find("Player");
        rb = GetComponent<Rigidbody2D>();
        maxHealth = health;
    }

    void Update() {
        timePenalty += (Time.deltaTime / 50);
        //set animator isBurning equal to this script's isBurning
        GetComponent<Animator>().SetBool("isBurning", isBurning);
        if(isBurning && burnSound.isPlaying == false) {
            burnSound.Play();
        } else if(!isBurning && burnSound.isPlaying == true) {
            burnSound.Stop();
        }
        //turn to always face the player
        Vector2 direction = (player.transform.position - transform.position).normalized;
        transform.up = direction;
        //set canvas to never rotate.  it is a child of this
        canvas.transform.localRotation = Quaternion.identity;

        //if is burning is true, take damage every second
        if(isBurning && Time.time - timeLastBurnedDamage > 1) {
            timeLastBurnedDamage = Time.time;
            TakeDamage(0.5f);
        }

        if(isSlowed && wetParticles.isPlaying == false) {
            wetParticles.Play();
        }
        else if(!isSlowed && wetParticles.isPlaying == true) {
            wetParticles.Stop();
        }
        if(Time.time - timeLastWaterHit > 0.2f) {
            hitByWaterCount = 0;
        }

        //if touching rock, take damage every 0.75 seconds
        if(touchingRock) {
            if(touchedRock.GetComponent<Earth>().isBurning && Time.time - timeLastDamagedRock > fireDamageDelay) {
                timeLastDamagedRock = Time.time;
                TakeDamage(1f);
            } else if(Time.time - timeLastDamagedRock > rockDamageDelay) {
                timeLastDamagedRock = Time.time;
                touchedRock.GetComponent<Earth>().TakeDamage(damage);
            }
        }

        if(nearPlayer) {
            if(Time.time - timeLastAttacked > attackRate) {
                timeLastAttacked = Time.time;
                if(player.GetComponent<Player>() == null)  
                {
                    player.GetComponent<PlayerRL>().TakeDamage(damage);
                } else 
                {
                    player.GetComponent<Player>().TakeDamage(damage);
                }
            }
        }

    }

    void FixedUpdate() {
        //if the enemy is outside a certain radius from the player, move towards the player
        if (Vector2.Distance(transform.position, player.transform.position) > 2f) {
            Vector2 direction = (player.transform.position - transform.position).normalized;
            if(isSlowed) {
                rb.velocity = direction * movementSpeed / 4f;
            } else {
                rb.velocity = direction * movementSpeed;
            }
            nearPlayer = false;
        } else {
            rb.velocity = Vector2.zero;
            nearPlayer = true;
            //we are close enough to the enemy to attack now
        }
        
    }

    public void TakeDamage(float damage) {
        health -= damage;
        if(player.GetComponent<PlayerRL>() != null)
        {
            // player.GetComponent<PlayerRL>().AddRewardCustom(1);
        }
        healthBar.fillAmount = health / maxHealth;
        if (health <= 0) {
            if(player.GetComponent<PlayerRL>() != null)
            {
                //UNCOMMENT FOR FIRE TRAINING
                // for(int i = 0; i < player.GetComponent<PlayerRL>().closestEnemies.Length; i++) {
                //     if(player.GetComponent<PlayerRL>().closestEnemies[i] == gameObject) {
                //         player.GetComponent<PlayerRL>().AddRewardExternal(1);
                //         //log distance to the player
                //         float distance = Vector2.Distance(player.transform.position, transform.position);
                //         if(distance < 3.1f)
                //         {
                //             //we will penalize the agent for killing an enemy too close to the player
                //             player.GetComponent<PlayerRL>().AddRewardExternal(-1);
                //         }
                //     }
                // }
                
            }
            Destroy(gameObject);
        }

    }

    public void SlowDown() {
        isSlowed = true;
        StartCoroutine(undoSlowDown());
    }

    public void ChainLightning() {
        //check in a given radius away, for fellow enemies. if any of them have a direct line of sight and are also isSlowed, then render a line between this enemy and that enemy. heres the code:




        Collider2D[] enemies = Physics2D.OverlapCircleAll(transform.position, 10f);
        foreach(Collider2D enemy in enemies) {
            if(enemy.gameObject.tag == "Enemy" && enemy.gameObject.GetComponent<Enemy>().isSlowed) {
                //draw a line between this enemy and that enemy
                //LineRenderer lr should be the component of an instantiation of chainLightningPrefab
                //instantiate a new line renderer between this enemy and that one. this should be a yellow line with good thickness,

                //and should be destroyed after 0.8 seconds. it should follow the positions of the two enemies. use a coroutine to do this.
                //create lr. do not use chainLightningPrefab, just create a new line renderer
                LineRenderer lr = Instantiate(chainLightningPrefab, transform.position, Quaternion.identity).GetComponent<LineRenderer>();
                lr.SetPosition(0, transform.position);
                lr.SetPosition(1, enemy.transform.position);
                //make it yellow with opacity 0.5. do it:
                lr.startColor = new Color(1f, 1f, 0f, 0.5f);
                lr.endColor = new Color(1f, 1f, 0f, 0.5f);
                lr.startWidth = 0.15f;
                lr.endWidth = 0.15f;
                //make opacity 0.5
                lr.material = new Material(Shader.Find("Sprites/Default"));
                //now call a coroutine to follow the two enemies
                //make the other enemy take damage of 2
                enemy.gameObject.GetComponent<Enemy>().TakeDamage(2f);
                //the lr has a script called ChainLightning. it has fields GameObject enemyOne and enemyTwo. set them equal to this enemy and that enemy
                if(lr != null) {
                    lr.GetComponent<ChainLightning>().enemyOne = gameObject;
                    lr.GetComponent<ChainLightning>().enemyTwo = enemy.gameObject;
                }

                

        
            }
        }
    }

    // IEnumerator followEnemies(LineRenderer lr, GameObject enemy) {
    //     //follow em, die after 0.8 seconds
    //     for(float i = 0; i < 80; i += 1) {
    //         lr.SetPosition(0, transform.position);
    //         lr.SetPosition(1, enemy.transform.position);
    //         yield return new WaitForSeconds(0.01f );
    //     }
    //     Destroy(lr.gameObject);
            

    // }

    IEnumerator undoSlowDown() {
        yield return new WaitForSeconds(5f);
        isSlowed = false;
    }

    //on collision enter of tag Earth, print yes!
    void OnCollisionEnter2D(Collision2D collision) {
        if(collision.gameObject.tag == "Earth") {
            //print("yes!");
            //take damage to earth
            touchedRock = collision.gameObject;
            touchingRock = true;
            timeLastDamagedRock = Time.time;
            
        } else if(collision.gameObject.tag == "Water") {

        }
    }
    //same idea for leaving collision, print no!
    void OnCollisionExit2D(Collision2D collision) {
        if(collision.gameObject.tag == "Earth") {
            touchingRock = false;
            touchedRock = null;
        }
    }
}
